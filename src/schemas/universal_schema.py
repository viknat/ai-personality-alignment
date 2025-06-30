"""
Data schemas for Universal User Embeddings training.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class UserChunk:
    """Represents a single chunk of user data."""
    text: str
    platform: str
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class UserData:
    """Represents a user's complete data history."""
    user_id: str
    chunks: List[UserChunk]
    statements: List[str]  # Positive statements about this user
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class StatementData:
    """Represents a behavioral statement with optional context."""
    statement: str
    context: Optional[str] = None
    category: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TrainingExample:
    """A training example with user data and positive/negative statements."""
    user_data: UserData
    positive_statement: str
    negative_statements: List[str]
    context: Optional[str] = None


@dataclass
class BatchData:
    """Batch data for training."""
    user_ids: List[str]
    user_texts: List[List[str]]  # List of chunks per user
    positive_statements: List[str]
    negative_statements: List[List[str]]  # List of negative statements per user
    contexts: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ModelConfig:
    """Configuration for the Universal User Embeddings model."""
    # Model architecture
    base_model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dim: int = 1024  # Qwen3-Embedding-0.6B output dimension
    max_chunk_length: int = 250
    max_statement_length: int = 128
    max_context_length: int = 64
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 2e-5
    temperature: float = 0.1
    num_epochs: int = 10
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    
    # Data processing
    num_chunks_per_user: int = 100
    num_negative_samples: int = 15
    chunk_overlap: int = 50
    
    # Architecture specific
    use_attention_pooling: bool = True
    attention_heads: int = 8
    dropout: float = 0.1
    
    # Logging and evaluation
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    eval_batch_size: int = 64