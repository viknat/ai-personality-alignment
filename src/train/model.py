"""
Universal User Embedding Model Architecture.

This implements the proper two-tower framework for recommendation systems:
- User Tower: Separate model for encoding user features/history
- Item Tower: Separate model for encoding item features/descriptions
- Both towers map to a shared embedding space for similarity computation
- No weight sharing between towers - each is optimized for its specific input type
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Optional, Tuple, Dict, Any, Union
import numpy as np

from ..schemas.universal_schema import ModelConfig


class UserTower(nn.Module):
    """
    User Tower - encodes user features and history into embeddings.
    
    This tower is specifically designed for user data:
    - User text history (posts, messages, reviews)
    - User metadata (age, location, preferences)
    - User behavior patterns
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # User-specific tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(config.user_model_name)
        self.transformer = AutoModel.from_pretrained(config.user_model_name)
        
        # Ensure tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # User-specific processing layers
        self.user_attention_pooling = nn.MultiheadAttention(
            embed_dim=self.transformer.config.hidden_size,
            num_heads=config.attention_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # User metadata processing (if available)
        if config.use_user_metadata:
            self.metadata_projection = nn.Linear(config.metadata_dim, self.transformer.config.hidden_size)
            self.metadata_gate = nn.Sequential(
                nn.Linear(self.transformer.config.hidden_size * 2, self.transformer.config.hidden_size),
                nn.Sigmoid()
            )
        
        # Final projection to shared embedding space
        self.user_projection = nn.Linear(self.transformer.config.hidden_size, config.embedding_dim)
        self.user_dropout = nn.Dropout(config.dropout)
        
    def forward(self, 
                user_chunks: List[List[str]], 
                user_metadata: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode user data into embeddings.
        
        Args:
            user_chunks: List of text chunks for each user
            user_metadata: Optional user metadata tensor (batch_size, metadata_dim)
            
        Returns:
            user_embeddings: (batch_size, embedding_dim) - L2 normalized
        """
        batch_size = len(user_chunks)
        max_chunks = max(len(chunks) for chunks in user_chunks)
        
        # Process text chunks
        all_chunk_embeddings = []
        attention_masks = []
        
        for user_chunks_list in user_chunks:
            # Pad chunks
            padded_chunks = user_chunks_list + [""] * (max_chunks - len(user_chunks_list))
            mask = [1] * len(user_chunks_list) + [0] * (max_chunks - len(user_chunks_list))
            attention_masks.append(mask)
            
            # Tokenize and encode
            tokenized = self.tokenizer(
                padded_chunks,
                padding=True,
                truncation=True,
                max_length=self.config.max_chunk_length,
                return_tensors="pt"
            )
            
            device = next(self.parameters()).device
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)
            
            with torch.no_grad():
                outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
                chunk_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            
            all_chunk_embeddings.append(chunk_embeddings)
        
        # Stack and apply attention pooling
        chunk_embeddings = torch.stack(all_chunk_embeddings)  # (batch_size, max_chunks, hidden_dim)
        attention_masks = torch.tensor(attention_masks, device=chunk_embeddings.device)
        
        # Apply user-specific attention pooling
        attended_embeddings, _ = self.user_attention_pooling(
            chunk_embeddings, chunk_embeddings, chunk_embeddings,
            key_padding_mask=(attention_masks == 0)
        )
        
        # Global pooling
        mask_expanded = attention_masks.unsqueeze(-1).float()
        user_embeddings = (attended_embeddings * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        
        # Incorporate metadata if available
        if user_metadata is not None and hasattr(self, 'metadata_projection'):
            metadata_embeddings = self.metadata_projection(user_metadata)
            gate = self.metadata_gate(torch.cat([user_embeddings, metadata_embeddings], dim=1))
            user_embeddings = gate * user_embeddings + (1 - gate) * metadata_embeddings
        
        # Project to shared embedding space
        user_embeddings = self.user_projection(user_embeddings)
        user_embeddings = self.user_dropout(user_embeddings)
        user_embeddings = F.normalize(user_embeddings, p=2, dim=1)
        
        return user_embeddings


class ItemTower(nn.Module):
    """
    Item Tower - encodes item features and descriptions into embeddings.
    
    This tower is specifically designed for item data:
    - Item descriptions, titles, categories
    - Item metadata (price, brand, features)
    - Item behavior patterns (views, clicks, purchases)
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Item-specific tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(config.item_model_name)
        self.transformer = AutoModel.from_pretrained(config.item_model_name)
        
        # Ensure tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Item-specific processing layers
        self.item_attention = nn.MultiheadAttention(
            embed_dim=self.transformer.config.hidden_size,
            num_heads=config.attention_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Item metadata processing (if available)
        if config.use_item_metadata:
            self.item_metadata_projection = nn.Linear(config.item_metadata_dim, self.transformer.config.hidden_size)
            self.item_metadata_gate = nn.Sequential(
                nn.Linear(self.transformer.config.hidden_size * 2, self.transformer.config.hidden_size),
                nn.Sigmoid()
            )
        
        # Final projection to shared embedding space
        self.item_projection = nn.Linear(self.transformer.config.hidden_size, config.embedding_dim)
        self.item_dropout = nn.Dropout(config.dropout)
        
    def forward(self, 
                item_texts: List[str], 
                item_metadata: Optional[torch.Tensor] = None,
                contexts: Optional[List[str]] = None) -> torch.Tensor:
        """
        Encode item data into embeddings.
        
        Args:
            item_texts: List of item descriptions/statements
            item_metadata: Optional item metadata tensor (batch_size, item_metadata_dim)
            contexts: Optional context strings for items
            
        Returns:
            item_embeddings: (batch_size, embedding_dim) - L2 normalized
        """
        # Combine texts with contexts if provided
        if contexts is not None:
            combined_texts = []
            for text, ctx in zip(item_texts, contexts):
                if ctx:
                    combined_texts.append(f"{ctx} {text}")
                else:
                    combined_texts.append(text)
        else:
            combined_texts = item_texts
        
        # Tokenize and encode
        tokenized = self.tokenizer(
            combined_texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_statement_length,
            return_tensors="pt"
        )
        
        device = next(self.parameters()).device
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)
        
        with torch.no_grad():
            outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
            item_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Apply item-specific attention if we have multiple features
        if item_metadata is not None and hasattr(self, 'item_metadata_projection'):
            metadata_embeddings = self.item_metadata_projection(item_metadata)
            gate = self.item_metadata_gate(torch.cat([item_embeddings, metadata_embeddings], dim=1))
            item_embeddings = gate * item_embeddings + (1 - gate) * metadata_embeddings
        
        # Project to shared embedding space
        item_embeddings = self.item_projection(item_embeddings)
        item_embeddings = self.item_dropout(item_embeddings)
        item_embeddings = F.normalize(item_embeddings, p=2, dim=1)
        
        return item_embeddings


class UniversalUserEmbeddingModel(nn.Module):
    """
    Universal User Embedding Model implementing proper two-tower architecture.
    
    This model consists of:
    1. Separate User Tower: Optimized for user data processing
    2. Separate Item Tower: Optimized for item data processing
    3. Both towers map to a shared embedding space
    4. No weight sharing - each tower is specialized for its input type
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Initialize separate towers
        self.user_tower = UserTower(config)
        self.item_tower = ItemTower(config)
        
        # Temperature parameter for contrastive learning with bounds
        # Use log-temperature to prevent gradient explosion: τ = exp(log_τ)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(config.temperature)))
        # Register bounds for gradient clipping: [log(0.01), log(10.0)]
        self.register_buffer('log_temp_min', torch.log(torch.tensor(0.01)))
        self.register_buffer('log_temp_max', torch.log(torch.tensor(10.0)))
    
    @property
    def temperature(self):
        """Get bounded temperature value."""
        log_temp = torch.clamp(self.log_temperature, self.log_temp_min, self.log_temp_max)
        return torch.exp(log_temp)
    
    def encode_chunks(self, chunks: List[List[str]], user_metadata: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode users using the user tower."""
        return self.user_tower(chunks, user_metadata)
    
    def encode_statements(self, statements: List[str], item_metadata: Optional[torch.Tensor] = None, contexts: Optional[List[str]] = None) -> torch.Tensor:
        """Encode items using the item tower."""
        return self.item_tower(statements, item_metadata, contexts)
    
    def forward(self, 
                user_chunks: List[List[str]], 
                positive_statements: List[str],
                negative_statements: Optional[List[List[str]]] = None,
                user_metadata: Optional[torch.Tensor] = None,
                item_metadata: Optional[torch.Tensor] = None,
                contexts: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            user_chunks: List of chunk texts for each user
            positive_statements: List of positive statements for each user
            negative_statements: Optional list of negative statements for each user
            user_metadata: Optional user metadata
            item_metadata: Optional item metadata
            contexts: Optional context strings for items
            
        Returns:
            Dictionary containing embeddings and similarity scores
        """
        # Encode users and positive items using separate towers
        user_embeddings = self.encode_chunks(user_chunks, user_metadata)
        positive_embeddings = self.encode_statements(positive_statements, item_metadata, contexts)
        
        # Compute positive similarities
        positive_similarities = torch.sum(user_embeddings * positive_embeddings, dim=1) / self.temperature
        
        result = {
            "user_embeddings": user_embeddings,
            "positive_embeddings": positive_embeddings,
            "positive_similarities": positive_similarities,
            "temperature": self.temperature
        }
        
        # Encode negative items if provided
        if negative_statements is not None:
            # Flatten negative items for batch processing
            flat_negatives = [stmt for user_negs in negative_statements for stmt in user_negs]
            negative_embeddings = self.encode_statements(flat_negatives, item_metadata, contexts)
            
            # Reshape back to (batch_size, num_negatives, embedding_dim)
            num_negatives = len(negative_statements[0])
            negative_embeddings = negative_embeddings.view(len(user_chunks), num_negatives, -1)
            
            # Compute negative similarities
            negative_similarities = torch.bmm(
                user_embeddings.unsqueeze(1), 
                negative_embeddings.transpose(1, 2)
            ).squeeze(1) / self.temperature
            
            result.update({
                "negative_embeddings": negative_embeddings,
                "negative_similarities": negative_similarities
            })
        
        return result
    
    def get_user_embedding(self, user_chunks: List[str], user_metadata: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get embedding for a single user."""
        self.eval()
        with torch.no_grad():
            embedding = self.encode_chunks([user_chunks], user_metadata)
            return embedding.squeeze(0)
    
    def get_statement_embedding(self, statement: str, item_metadata: Optional[torch.Tensor] = None, context: Optional[str] = None) -> torch.Tensor:
        """Get embedding for a single statement."""
        self.eval()
        with torch.no_grad():
            context_list = [context] if context else None
            embedding = self.encode_statements([statement], item_metadata, context_list)
            return embedding.squeeze(0) 