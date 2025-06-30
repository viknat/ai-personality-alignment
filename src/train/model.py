"""
Universal User Embedding Model Architecture.

This implements the dual-encoder framework described in the paper:
- User Encoder: Processes user history chunks with attention pooling
- Statement Encoder: Processes behavioral statements with optional context
- Both encoders share weights and produce normalized embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from ..schemas.universal_schema import ModelConfig


class AttentionPooling(nn.Module):
    """Attention-based pooling mechanism for aggregating chunk representations."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, chunk_embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            chunk_embeddings: (batch_size, num_chunks, hidden_dim)
            attention_mask: (batch_size, num_chunks) - 1 for valid chunks, 0 for padding
            
        Returns:
            pooled_embedding: (batch_size, hidden_dim)
        """
        batch_size, num_chunks, hidden_dim = chunk_embeddings.shape
        
        # Multi-head attention
        Q = self.query(chunk_embeddings).view(batch_size, num_chunks, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(chunk_embeddings).view(batch_size, num_chunks, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(chunk_embeddings).view(batch_size, num_chunks, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask for multi-head attention
            mask = attention_mask.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, num_chunks)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, num_chunks, hidden_dim)
        
        # Project output
        output = self.output_proj(attended)
        
        # Global pooling: take mean across chunks
        if attention_mask is not None:
            # Weighted average based on attention mask
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (output * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            pooled = output.mean(dim=1)
            
        return pooled


class UniversalUserEmbeddingModel(nn.Module):
    """
    Universal User Embedding Model implementing the dual-encoder architecture.
    
    This model consists of:
    1. A shared transformer encoder for both user chunks and statements
    2. An attention pooling mechanism for aggregating user chunk representations
    3. L2 normalization for both user and statement embeddings
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Load the base Qwen3 embedding model
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
        self.transformer = AutoModel.from_pretrained(config.base_model_name)
        
        # Ensure the tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Attention pooling for user chunks
        if config.use_attention_pooling:
            self.attention_pooling = AttentionPooling(
                hidden_dim=self.transformer.config.hidden_size,
                num_heads=config.attention_heads,
                dropout=config.dropout
            )
        else:
            self.attention_pooling = None
            
        # Projection layer to match embedding dimension
        if self.transformer.config.hidden_size != config.embedding_dim:
            self.projection = nn.Linear(self.transformer.config.hidden_size, config.embedding_dim)
        else:
            self.projection = None
            
        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout)
        
    def encode_chunks(self, chunks: List[List[str]], max_length: Optional[int] = None) -> torch.Tensor:
        """
        Encode user chunks into embeddings.
        
        Args:
            chunks: List of chunk texts for each user in the batch
            max_length: Maximum sequence length for tokenization
            
        Returns:
            user_embeddings: (batch_size, embedding_dim) - L2 normalized
        """
        if max_length is None:
            max_length = self.config.max_chunk_length
            
        batch_size = len(chunks)
        max_chunks = max(len(user_chunks) for user_chunks in chunks)
        
        # Pad chunks to same length
        padded_chunks = []
        attention_masks = []
        
        for user_chunks in chunks:
            # Pad with empty strings if needed
            padded_user_chunks = user_chunks + [""] * (max_chunks - len(user_chunks))
            padded_chunks.append(padded_user_chunks)
            
            # Create attention mask
            mask = [1] * len(user_chunks) + [0] * (max_chunks - len(user_chunks))
            attention_masks.append(mask)
        
        # Flatten for batch processing
        flat_chunks = [chunk for user_chunks in padded_chunks for chunk in user_chunks]
        
        # Tokenize
        tokenized = self.tokenizer(
            flat_chunks,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Move to device
        device = next(self.parameters()).device
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)
        
        # Get embeddings from transformer
        with torch.no_grad():
            outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
            chunk_embeddings = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        
        # Reshape back to (batch_size, max_chunks, hidden_dim)
        chunk_embeddings = chunk_embeddings.view(batch_size, max_chunks, -1)
        attention_masks = torch.tensor(attention_masks, device=device)
        
        # Apply attention pooling if enabled
        if self.attention_pooling is not None:
            user_embeddings = self.attention_pooling(chunk_embeddings, attention_masks)
        else:
            # Simple mean pooling
            mask_expanded = attention_masks.unsqueeze(-1).float()
            user_embeddings = (chunk_embeddings * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        
        # Apply projection if needed
        if self.projection is not None:
            user_embeddings = self.projection(user_embeddings)
            
        # Apply dropout and normalize
        user_embeddings = self.dropout(user_embeddings)
        user_embeddings = F.normalize(user_embeddings, p=2, dim=1)
        
        return user_embeddings
    
    def encode_statements(self, statements: List[str], contexts: Optional[List[str]] = None) -> torch.Tensor:
        """
        Encode behavioral statements into embeddings.
        
        Args:
            statements: List of behavioral statements
            contexts: Optional list of context strings
            
        Returns:
            statement_embeddings: (batch_size, embedding_dim) - L2 normalized
        """
        # Combine statements with contexts if provided
        if contexts is not None:
            combined_texts = []
            for stmt, ctx in zip(statements, contexts):
                if ctx:
                    combined_texts.append(f"{ctx} {stmt}")
                else:
                    combined_texts.append(stmt)
        else:
            combined_texts = statements
            
        # Tokenize
        tokenized = self.tokenizer(
            combined_texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_statement_length,
            return_tensors="pt"
        )
        
        # Move to device
        device = next(self.parameters()).device
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)
        
        # Get embeddings from transformer
        with torch.no_grad():
            outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
            statement_embeddings = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        
        # Apply projection if needed
        if self.projection is not None:
            statement_embeddings = self.projection(statement_embeddings)
            
        # Apply dropout and normalize
        statement_embeddings = self.dropout(statement_embeddings)
        statement_embeddings = F.normalize(statement_embeddings, p=2, dim=1)
        
        return statement_embeddings
    
    def forward(self, 
                user_chunks: List[List[str]], 
                positive_statements: List[str],
                negative_statements: Optional[List[List[str]]] = None,
                contexts: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            user_chunks: List of chunk texts for each user
            positive_statements: List of positive statements for each user
            negative_statements: Optional list of negative statements for each user
            contexts: Optional list of context strings
            
        Returns:
            Dictionary containing embeddings and similarity scores
        """
        # Encode users and positive statements
        user_embeddings = self.encode_chunks(user_chunks)
        positive_embeddings = self.encode_statements(positive_statements, contexts)
        
        # Compute positive similarities
        positive_similarities = torch.sum(user_embeddings * positive_embeddings, dim=1)
        
        result = {
            "user_embeddings": user_embeddings,
            "positive_embeddings": positive_embeddings,
            "positive_similarities": positive_similarities
        }
        
        # Encode negative statements if provided
        if negative_statements is not None:
            # Flatten negative statements for batch processing
            flat_negatives = [stmt for user_negs in negative_statements for stmt in user_negs]
            negative_embeddings = self.encode_statements(flat_negatives)
            
            # Reshape back to (batch_size, num_negatives, embedding_dim)
            num_negatives = len(negative_statements[0])
            negative_embeddings = negative_embeddings.view(len(user_chunks), num_negatives, -1)
            
            # Compute negative similarities
            negative_similarities = torch.bmm(
                user_embeddings.unsqueeze(1), 
                negative_embeddings.transpose(1, 2)
            ).squeeze(1)
            
            result.update({
                "negative_embeddings": negative_embeddings,
                "negative_similarities": negative_similarities
            })
        
        return result
    
    def get_user_embedding(self, user_chunks: List[str]) -> torch.Tensor:
        """
        Get embedding for a single user.
        
        Args:
            user_chunks: List of text chunks for the user
            
        Returns:
            user_embedding: (embedding_dim,) - L2 normalized
        """
        self.eval()
        with torch.no_grad():
            embedding = self.encode_chunks([user_chunks])
            return embedding.squeeze(0)
    
    def get_statement_embedding(self, statement: str, context: Optional[str] = None) -> torch.Tensor:
        """
        Get embedding for a single statement.
        
        Args:
            statement: Behavioral statement
            context: Optional context string
            
        Returns:
            statement_embedding: (embedding_dim,) - L2 normalized
        """
        self.eval()
        with torch.no_grad():
            context_list = [context] if context else None
            embedding = self.encode_statements([statement], context_list)
            return embedding.squeeze(0) 