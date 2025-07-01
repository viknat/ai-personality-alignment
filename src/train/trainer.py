"""
Universal User Embeddings Trainer.

This module implements the complete training pipeline for the Universal User Embeddings model,
including training loops, evaluation, logging, and model saving.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.utils.data import DataLoader
import wandb
import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
import logging
from pathlib import Path

from .model import UniversalUserEmbeddingModel
from .loss import ContrastiveLoss, HardNegativeMiningLoss, TripletLoss
from .data_loader import UserEmbeddingDataset, UserDataLoader, create_sample_data, create_evaluation_data
from ..schemas.universal_schema import ModelConfig, UserData, BatchData


class UniversalUserEmbeddingTrainer:
    """
    Trainer for Universal User Embeddings model with proper two-tower architecture.
    
    This class handles:
    - Model training with contrastive learning using separate optimizers
    - Evaluation and metrics tracking
    - Model checkpointing and saving
    - Logging with wandb
    - Learning rate scheduling for each tower
    """
    
    def __init__(self, 
                 config: ModelConfig,
                 model: Optional[UniversalUserEmbeddingModel] = None,
                 train_data: Optional[List[UserData]] = None,
                 eval_data: Optional[List[UserData]] = None,
                 statement_pool: Optional[List[str]] = None,
                 output_dir: str = "./outputs",
                 use_wandb: bool = True,
                 use_mixed_precision: bool = True):
        """
        Initialize the trainer.
        
        Args:
            config: Model configuration
            model: Pre-initialized model (optional)
            train_data: Training data (optional)
            eval_data: Evaluation data (optional)
            statement_pool: Pool of statements for negative sampling
            output_dir: Directory to save outputs
            use_wandb: Whether to use wandb for logging
            use_mixed_precision: Whether to use mixed precision training
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        self.use_mixed_precision = use_mixed_precision
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize model
        if model is None:
            self.model = UniversalUserEmbeddingModel(config)
        else:
            self.model = model
        self.model.to(self.device)
        
        # Initialize loss function
        self.loss_fn = ContrastiveLoss(
            temperature=config.temperature,
            learnable_temperature=True  # Enable learnable temperature
        ).to(self.device)
        
        # Initialize separate optimizers for each tower
        self.user_optimizer = optim.AdamW(
            self.model.user_tower.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.item_optimizer = optim.AdamW(
            self.model.item_tower.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Initialize separate schedulers
        self.user_scheduler = CosineAnnealingLR(
            self.user_optimizer,
            T_max=config.num_epochs,
            eta_min=config.learning_rate * 0.01
        )
        
        self.item_scheduler = CosineAnnealingLR(
            self.item_optimizer,
            T_max=config.num_epochs,
            eta_min=config.learning_rate * 0.01
        )
        
        # Setup mixed precision training
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Setup data
        if train_data is None or statement_pool is None:
            print("Creating sample data for demonstration...")
            train_data, statement_pool = create_sample_data(config)
        
        if eval_data is None:
            print("Creating evaluation data...")
            eval_data, _ = create_evaluation_data(config)
        
        self.train_data = train_data
        self.eval_data = eval_data
        self.statement_pool = statement_pool
        
        # Create datasets and dataloaders
        self.train_dataset = UserEmbeddingDataset(
            train_data, statement_pool, config, negative_sampling_strategy="random"
        )
        self.eval_dataset = UserEmbeddingDataset(
            eval_data, statement_pool, config, negative_sampling_strategy="random"
        )
        
        self.train_loader = UserDataLoader(
            self.train_dataset, config, shuffle=True, num_workers=2
        )
        self.eval_loader = UserDataLoader(
            self.eval_dataset, config, shuffle=False, num_workers=2
        )
        
        # Setup logging
        self.setup_logging()
        
        # Training state
        self.global_step = 0
        self.best_eval_loss = float('inf')
        self.best_eval_acc = 0.0
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / "two_tower_training.log"),
                logging.StreamHandler()
            ]
        )
        
        if self.use_wandb:
            wandb.init(
                project="two-tower-user-embeddings",
                config=vars(self.config),
                name=f"two_tower_{self.config.user_model_name.split('/')[-1]}_{self.config.item_model_name.split('/')[-1]}"
            )
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch with proper two-tower optimization.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}",
            leave=False
        )
        
        for batch_idx, batch_data in enumerate(progress_bar):
            # Forward pass with mixed precision
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        user_chunks=batch_data.user_texts,
                        positive_statements=batch_data.positive_statements,
                        negative_statements=batch_data.negative_statements
                    )
                    
                    # Compute loss
                    loss_result = self.loss_fn(
                        user_embeddings=outputs["user_embeddings"],
                        positive_embeddings=outputs["positive_embeddings"],
                        negative_embeddings=outputs.get("negative_embeddings")
                    )
                    
                    loss = loss_result["loss"]
            else:
                outputs = self.model(
                    user_chunks=batch_data.user_texts,
                    positive_statements=batch_data.positive_statements,
                    negative_statements=batch_data.negative_statements
                )
                
                # Compute loss
                loss_result = self.loss_fn(
                    user_embeddings=outputs["user_embeddings"],
                    positive_embeddings=outputs["positive_embeddings"],
                    negative_embeddings=outputs.get("negative_embeddings")
                )
                
                loss = loss_result["loss"]
            
            # Backward pass with separate optimizers
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping for user tower
                self.scaler.unscale_(self.user_optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.user_tower.parameters(), max_norm=1.0)
                
                # Gradient clipping for item tower
                self.scaler.unscale_(self.item_optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.item_tower.parameters(), max_norm=1.0)
                
                # Gradient clipping for temperature parameter (prevent explosion)
                if hasattr(self.model, 'log_temperature'):
                    torch.nn.utils.clip_grad_norm_([self.model.log_temperature], max_norm=0.1)
                if hasattr(self.loss_fn, 'log_temperature') and self.loss_fn.log_temperature is not None:
                    torch.nn.utils.clip_grad_norm_([self.loss_fn.log_temperature], max_norm=0.1)
                
                # Step optimizers
                self.scaler.step(self.user_optimizer)
                self.scaler.step(self.item_optimizer)
                self.scaler.update()
            else:
                # Regular backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.user_tower.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.model.item_tower.parameters(), max_norm=1.0)
                
                # Gradient clipping for temperature parameter (prevent explosion)
                if hasattr(self.model, 'log_temperature'):
                    torch.nn.utils.clip_grad_norm_([self.model.log_temperature], max_norm=0.1)
                if hasattr(self.loss_fn, 'log_temperature') and self.loss_fn.log_temperature is not None:
                    torch.nn.utils.clip_grad_norm_([self.loss_fn.log_temperature], max_norm=0.1)
                
                # Step optimizers
                self.user_optimizer.step()
                self.item_optimizer.step()
            
            # Zero gradients
            self.user_optimizer.zero_grad()
            self.item_optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item()
            total_acc += loss_result["avg_acc"].item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{loss_result['avg_acc'].item():.4f}",
                'temp': f"{outputs['temperature'].item():.3f}"
            })
            
            # Log to wandb
            if self.use_wandb and batch_idx % self.config.logging_steps == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/accuracy": loss_result["avg_acc"].item(),
                    "train/temperature": outputs["temperature"].item(),
                    "train/user_lr": self.user_optimizer.param_groups[0]["lr"],
                    "train/item_lr": self.item_optimizer.param_groups[0]["lr"],
                    "train/step": self.global_step
                })
            
            self.global_step += 1
            
            # Evaluation
            if self.global_step % self.config.eval_steps == 0:
                eval_metrics = self.evaluate()
                self.log_eval_metrics(eval_metrics)
                
                # Save best model
                if eval_metrics["eval/accuracy"] > self.best_eval_acc:
                    self.best_eval_acc = eval_metrics["eval/accuracy"]
                    self.save_model("best_two_tower_model.pt")
            
            # Save checkpoint
            if self.global_step % self.config.save_steps == 0:
                self.save_checkpoint(f"two_tower_checkpoint_step_{self.global_step}.pt")
        
        # Update learning rate schedulers
        self.user_scheduler.step()
        self.item_scheduler.step()
        
        return {
            "train/loss": total_loss / num_batches,
            "train/accuracy": total_acc / num_batches
        }
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the two-tower model on validation data.
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in self.eval_loader:
                # Forward pass
                outputs = self.model(
                    user_chunks=batch_data.user_texts,
                    positive_statements=batch_data.positive_statements,
                    negative_statements=batch_data.negative_statements
                )
                
                # Compute loss
                loss_result = self.loss_fn(
                    user_embeddings=outputs["user_embeddings"],
                    positive_embeddings=outputs["positive_embeddings"],
                    negative_embeddings=outputs.get("negative_embeddings")
                )
                
                total_loss += loss_result["loss"].item()
                total_acc += loss_result["avg_acc"].item()
                num_batches += 1
        
        return {
            "eval/loss": total_loss / num_batches,
            "eval/accuracy": total_acc / num_batches
        }
    
    def log_eval_metrics(self, metrics: Dict[str, float]):
        """Log evaluation metrics."""
        logging.info(f"Two-Tower Evaluation metrics: {metrics}")
        
        if self.use_wandb:
            wandb.log(metrics, step=self.global_step)
    
    def save_model(self, filename: str):
        """Save the two-tower model."""
        model_path = self.output_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "global_step": self.global_step,
            "best_eval_acc": self.best_eval_acc
        }, model_path)
        logging.info(f"Two-Tower model saved to {model_path}")
    
    def save_checkpoint(self, filename: str):
        """Save a training checkpoint for two-tower model."""
        checkpoint_path = self.output_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "user_optimizer_state_dict": self.user_optimizer.state_dict(),
            "item_optimizer_state_dict": self.item_optimizer.state_dict(),
            "user_scheduler_state_dict": self.user_scheduler.state_dict(),
            "item_scheduler_state_dict": self.item_scheduler.state_dict(),
            "config": self.config,
            "global_step": self.global_step,
            "best_eval_acc": self.best_eval_acc,
            "best_eval_loss": self.best_eval_loss
        }, checkpoint_path)
        logging.info(f"Two-Tower checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load a training checkpoint for two-tower model."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.user_optimizer.load_state_dict(checkpoint["user_optimizer_state_dict"])
        self.item_optimizer.load_state_dict(checkpoint["item_optimizer_state_dict"])
        self.user_scheduler.load_state_dict(checkpoint["user_scheduler_state_dict"])
        self.item_scheduler.load_state_dict(checkpoint["item_scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_eval_acc = checkpoint["best_eval_acc"]
        self.best_eval_loss = checkpoint["best_eval_loss"]
        
        logging.info(f"Two-Tower checkpoint loaded from {checkpoint_path}")
    
    def train(self):
        """Main training loop for two-tower model."""
        logging.info("Starting Two-Tower training...")
        logging.info(f"User Tower: {self.config.user_model_name}")
        logging.info(f"Item Tower: {self.config.item_model_name}")
        logging.info(f"Training on {len(self.train_data)} users")
        logging.info(f"Evaluating on {len(self.eval_data)} users")
        
        for epoch in range(self.config.num_epochs):
            # Train for one epoch
            train_metrics = self.train_epoch(epoch)
            
            # Evaluate
            eval_metrics = self.evaluate()
            
            # Log metrics
            all_metrics = {**train_metrics, **eval_metrics}
            logging.info(f"Epoch {epoch + 1} - {all_metrics}")
            
            if self.use_wandb:
                wandb.log(all_metrics, step=self.global_step)
            
            # Save best model
            if eval_metrics["eval/accuracy"] > self.best_eval_acc:
                self.best_eval_acc = eval_metrics["eval/accuracy"]
                self.save_model("best_two_tower_model.pt")
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f"two_tower_epoch_{epoch + 1}.pt")
        
        logging.info("Two-Tower training completed!")
        logging.info(f"Best evaluation accuracy: {self.best_eval_acc:.4f}")
    
    def test_user_embedding(self, user_chunks: List[str], user_metadata: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Test the two-tower model by getting a user embedding.
        
        Args:
            user_chunks: List of text chunks for the user
            user_metadata: Optional user metadata
            
        Returns:
            User embedding vector
        """
        self.model.eval()
        with torch.no_grad():
            embedding = self.model.get_user_embedding(user_chunks, user_metadata)
        return embedding
    
    def test_statement_embedding(self, statement: str, item_metadata: Optional[torch.Tensor] = None, context: Optional[str] = None) -> torch.Tensor:
        """
        Test the two-tower model by getting an item embedding.
        
        Args:
            statement: Item description/statement
            item_metadata: Optional item metadata
            context: Optional context string
            
        Returns:
            Item embedding vector
        """
        self.model.eval()
        with torch.no_grad():
            embedding = self.model.get_statement_embedding(statement, item_metadata, context)
        return embedding
    
    def compute_similarity(self, user_chunks: List[str], statement: str, 
                          user_metadata: Optional[torch.Tensor] = None,
                          item_metadata: Optional[torch.Tensor] = None,
                          context: Optional[str] = None) -> float:
        """
        Compute similarity between a user and an item using the two-tower model.
        
        Args:
            user_chunks: List of text chunks for the user
            statement: Item description/statement
            user_metadata: Optional user metadata
            item_metadata: Optional item metadata
            context: Optional context string
            
        Returns:
            Similarity score (cosine similarity)
        """
        user_embedding = self.test_user_embedding(user_chunks, user_metadata)
        statement_embedding = self.test_statement_embedding(statement, item_metadata, context)
        
        similarity = torch.sum(user_embedding * statement_embedding).item()
        return similarity


def main():
    """Main function to run two-tower training."""
    # Create configuration for proper two-tower model
    config = ModelConfig(
        user_model_name="Qwen/Qwen3-Embedding-0.6B",
        item_model_name="Qwen/Qwen3-Embedding-0.6B",
        embedding_dim=1024,
        batch_size=32,
        learning_rate=2e-5,
        temperature=0.1,
        num_epochs=10,
        use_user_metadata=True,
        use_item_metadata=True
    )
    
    # Initialize trainer
    trainer = UniversalUserEmbeddingTrainer(
        config=config,
        output_dir="./two_tower_outputs",
        use_wandb=True,
        use_mixed_precision=True
    )
    
    # Train the model
    trainer.train()
    
    # Test the model
    test_user_chunks = ["I love programming", "Working on AI projects"]
    test_item = "interested in technology"
    
    similarity = trainer.compute_similarity(test_user_chunks, test_item)
    print(f"Similarity between user and item: {similarity:.4f}")


if __name__ == "__main__":
    main() 