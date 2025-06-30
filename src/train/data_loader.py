"""
Data Loader for Universal User Embeddings Training.

This module handles data loading, preprocessing, and batch creation for training
the Universal User Embeddings model.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional, Tuple
import random
import numpy as np
from tqdm import tqdm

from ..schemas.universal_schema import UserData, StatementData, TrainingExample, BatchData, ModelConfig, UserChunk


class UserEmbeddingDataset(Dataset):
    """
    Dataset for Universal User Embeddings training.
    
    This dataset handles:
    - User data with multiple chunks
    - Positive and negative statement sampling
    - Context conditioning
    - Data augmentation
    """
    
    def __init__(self, 
                 user_data: List[UserData],
                 statement_pool: List[str],
                 config: ModelConfig,
                 negative_sampling_strategy: str = "random"):
        """
        Initialize the dataset.
        
        Args:
            user_data: List of user data objects
            statement_pool: Pool of statements for negative sampling
            config: Model configuration
            negative_sampling_strategy: Strategy for negative sampling ("random", "hard", "semi_hard")
        """
        self.user_data = user_data
        self.statement_pool = statement_pool
        self.config = config
        self.negative_sampling_strategy = negative_sampling_strategy
        
        # Create user ID to index mapping
        self.user_to_idx = {user.user_id: idx for idx, user in enumerate(user_data)}
        
        # Preprocess user chunks
        self._preprocess_user_chunks()
        
        # Create statement embeddings for negative sampling (if using hard negative mining)
        if negative_sampling_strategy in ["hard", "semi_hard"]:
            self._precompute_statement_embeddings()
    
    def _preprocess_user_chunks(self):
        """Preprocess user chunks to ensure consistent format."""
        for user in self.user_data:
            # Filter out empty chunks
            user.chunks = [chunk for chunk in user.chunks if chunk.text.strip()]
            
            # Limit number of chunks per user
            if len(user.chunks) > self.config.num_chunks_per_user:
                # Randomly sample chunks
                user.chunks = random.sample(user.chunks, self.config.num_chunks_per_user)
            
            # Ensure we have at least one chunk
            if not user.chunks:
                # Create a placeholder chunk
                user.chunks = [UserChunk(text="[EMPTY_USER]", platform="unknown")]
    
    def _precompute_statement_embeddings(self):
        """Precompute statement embeddings for hard negative mining."""
        # This would require a pre-trained model to compute embeddings
        # For now, we'll use a simple approach with random embeddings
        # In practice, you'd want to use a pre-trained sentence transformer
        self.statement_embeddings = {}
        for statement in self.statement_pool:
            # Simple hash-based embedding for demonstration
            # In practice, use a proper embedding model
            embedding = np.random.randn(self.config.embedding_dim)
            embedding = embedding / np.linalg.norm(embedding)
            self.statement_embeddings[statement] = embedding
    
    def _sample_negative_statements(self, 
                                  positive_statement: str, 
                                  user_id: str,
                                  num_negatives: int) -> List[str]:
        """
        Sample negative statements for a user.
        
        Args:
            positive_statement: The positive statement for this user
            user_id: User ID to avoid sampling their own statements
            num_negatives: Number of negative statements to sample
            
        Returns:
            List of negative statements
        """
        # Get all statements from other users
        other_user_statements = []
        for user in self.user_data:
            if user.user_id != user_id:
                other_user_statements.extend(user.statements)
        
        # Remove duplicates and the positive statement
        other_user_statements = list(set(other_user_statements))
        if positive_statement in other_user_statements:
            other_user_statements.remove(positive_statement)
        
        # Add statements from the general pool
        all_negative_candidates = other_user_statements + self.statement_pool
        
        if self.negative_sampling_strategy == "random":
            # Random sampling
            negatives = random.sample(all_negative_candidates, min(num_negatives, len(all_negative_candidates)))
            
        elif self.negative_sampling_strategy == "hard":
            # Hard negative mining (simplified version)
            # In practice, you'd compute similarities with the positive statement
            # and select the most similar ones as hard negatives
            negatives = random.sample(all_negative_candidates, min(num_negatives, len(all_negative_candidates)))
            
        elif self.negative_sampling_strategy == "semi_hard":
            # Semi-hard negative mining
            # Select negatives that are moderately similar to the positive
            negatives = random.sample(all_negative_candidates, min(num_negatives, len(all_negative_candidates)))
            
        else:
            raise ValueError(f"Unknown negative sampling strategy: {self.negative_sampling_strategy}")
        
        # Pad with random statements if we don't have enough
        while len(negatives) < num_negatives:
            negatives.append(random.choice(self.statement_pool))
        
        return negatives[:num_negatives]
    
    def __len__(self) -> int:
        """Return the number of training examples."""
        total_examples = 0
        for user in self.user_data:
            total_examples += len(user.statements)
        return total_examples
    
    def __getitem__(self, idx: int) -> TrainingExample:
        """
        Get a training example.
        
        Args:
            idx: Index of the training example
            
        Returns:
            TrainingExample object
        """
        # Find the user and statement for this index
        current_idx = 0
        for user in self.user_data:
            for statement in user.statements:
                if current_idx == idx:
                    # Sample negative statements
                    negative_statements = self._sample_negative_statements(
                        statement, user.user_id, self.config.num_negative_samples
                    )
                    
                    return TrainingExample(
                        user_data=user,
                        positive_statement=statement,
                        negative_statements=negative_statements
                    )
                current_idx += 1
        
        raise IndexError(f"Index {idx} out of range")
    
    def get_batch_data(self, batch_indices: List[int]) -> BatchData:
        """
        Get batch data for training.
        
        Args:
            batch_indices: List of indices for the batch
            
        Returns:
            BatchData object
        """
        user_ids = []
        user_texts = []
        positive_statements = []
        negative_statements = []
        
        for idx in batch_indices:
            example = self[idx]
            
            user_ids.append(example.user_data.user_id)
            user_texts.append([chunk.text for chunk in example.user_data.chunks])
            positive_statements.append(example.positive_statement)
            negative_statements.append(example.negative_statements)
        
        return BatchData(
            user_ids=user_ids,
            user_texts=user_texts,
            positive_statements=positive_statements,
            negative_statements=negative_statements
        )


class UserDataLoader:
    """
    Data loader for Universal User Embeddings training.
    
    This class handles batch creation, data collation, and provides
    convenient iteration over training data.
    """
    
    def __init__(self, 
                 dataset: UserEmbeddingDataset,
                 config: ModelConfig,
                 shuffle: bool = True,
                 num_workers: int = 4):
        """
        Initialize the data loader.
        
        Args:
            dataset: UserEmbeddingDataset instance
            config: Model configuration
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes
        """
        self.dataset = dataset
        self.config = config
        self.shuffle = shuffle
        self.num_workers = num_workers
        
        # Create PyTorch DataLoader
        self.dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True
        )
    
    def _collate_fn(self, batch: List[TrainingExample]) -> BatchData:
        """
        Collate function for batching training examples.
        
        Args:
            batch: List of TrainingExample objects
            
        Returns:
            BatchData object
        """
        user_ids = []
        user_texts = []
        positive_statements = []
        negative_statements = []
        
        for example in batch:
            user_ids.append(example.user_data.user_id)
            user_texts.append([chunk.text for chunk in example.user_data.chunks])
            positive_statements.append(example.positive_statement)
            negative_statements.append(example.negative_statements)
        
        return BatchData(
            user_ids=user_ids,
            user_texts=user_texts,
            positive_statements=positive_statements,
            negative_statements=negative_statements
        )
    
    def __iter__(self):
        """Iterate over batches."""
        return iter(self.dataloader)
    
    def __len__(self) -> int:
        """Return the number of batches."""
        return len(self.dataloader)


def create_sample_data(config: ModelConfig) -> Tuple[List[UserData], List[str]]:
    """
    Create sample data for testing and demonstration.
    
    Args:
        config: Model configuration
        
    Returns:
        Tuple of (user_data, statement_pool)
    """
    # Sample user data
    user_data = []
    
    # Sample users with different characteristics
    sample_users = [
        {
            "user_id": "user_001",
            "platform": "reddit",
            "chunks": [
                "I love discussing machine learning and AI. The latest developments in transformer models are fascinating.",
                "Just finished reading a paper on contrastive learning. The InfoNCE loss is really elegant.",
                "Working on a new project involving user embeddings. It's challenging but rewarding."
            ],
            "statements": [
                "interested in machine learning",
                "enjoys reading research papers",
                "works on AI projects",
                "discusses technical topics online"
            ]
        },
        {
            "user_id": "user_002", 
            "platform": "whatsapp",
            "chunks": [
                "Going to the gym later today. Need to work on my fitness goals.",
                "Had a great workout session yesterday. Feeling energized!",
                "Planning my meals for the week. Trying to eat healthier."
            ],
            "statements": [
                "interested in fitness and health",
                "goes to the gym regularly",
                "plans meals carefully",
                "focuses on personal wellness"
            ]
        },
        {
            "user_id": "user_003",
            "platform": "twitter", 
            "chunks": [
                "Just watched an amazing movie! The cinematography was incredible.",
                "Reading a new book by my favorite author. Can't put it down.",
                "Attended a great concert last night. The music was phenomenal."
            ],
            "statements": [
                "enjoys watching movies",
                "loves reading books",
                "attends concerts and live events",
                "appreciates arts and entertainment"
            ]
        }
    ]
    
    for user_info in sample_users:
        chunks = []
        for chunk_text in user_info["chunks"]:
            chunks.append(UserChunk(
                text=chunk_text,
                platform=user_info["platform"]
            ))
        
        user_data.append(UserData(
            user_id=user_info["user_id"],
            chunks=chunks,
            statements=user_info["statements"]
        ))
    
    # Sample statement pool
    statement_pool = [
        "likes cooking",
        "enjoys traveling",
        "plays musical instruments", 
        "interested in photography",
        "loves outdoor activities",
        "enjoys video games",
        "interested in politics",
        "loves animals",
        "enjoys gardening",
        "interested in history",
        "loves coffee",
        "enjoys hiking",
        "interested in fashion",
        "loves music",
        "enjoys painting"
    ]
    
    return user_data, statement_pool


def create_evaluation_data(config: ModelConfig) -> Tuple[List[UserData], List[str]]:
    """
    Create evaluation data for testing model performance.
    
    Args:
        config: Model configuration
        
    Returns:
        Tuple of (eval_user_data, eval_statements)
    """
    # Create evaluation users with known characteristics
    eval_user_data = []
    
    eval_users = [
        {
            "user_id": "eval_001",
            "platform": "reddit",
            "chunks": [
                "I'm a software engineer who loves coding in Python and JavaScript.",
                "Working on a new web application using React and Node.js.",
                "Just deployed my latest project to production. Feeling accomplished!"
            ],
            "statements": [
                "works as a software engineer",
                "programs in Python",
                "uses JavaScript for development",
                "builds web applications"
            ]
        },
        {
            "user_id": "eval_002",
            "platform": "instagram",
            "chunks": [
                "Beautiful sunset at the beach today! Nature is amazing.",
                "Just finished a 10-mile hike. The views were incredible.",
                "Camping trip this weekend. Can't wait to be outdoors!"
            ],
            "statements": [
                "enjoys outdoor activities",
                "loves nature and hiking",
                "goes camping regularly",
                "appreciates natural beauty"
            ]
        }
    ]
    
    for user_info in eval_users:
        chunks = []
        for chunk_text in user_info["chunks"]:
            chunks.append(UserChunk(
                text=chunk_text,
                platform=user_info["platform"]
            ))
        
        eval_user_data.append(UserData(
            user_id=user_info["user_id"],
            chunks=chunks,
            statements=user_info["statements"]
        ))
    
    # Evaluation statements
    eval_statements = [
        "works in technology",
        "enjoys outdoor recreation",
        "loves programming",
        "appreciates nature",
        "builds software",
        "goes hiking",
        "uses Python",
        "enjoys camping"
    ]
    
    return eval_user_data, eval_statements 