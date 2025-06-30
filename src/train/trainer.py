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
    Trainer for Universal User Embeddings model.
    
    This class handles:
    - Model training with contrastive learning
    - Evaluation and metrics tracking
    - Model checkpointing and saving
    - Logging with wandb
    - Learning rate scheduling
    """
    
    def __init__(self, 
                 config: ModelConfig,
                 model: Optional[UniversalUserEmbeddingModel] = None,
                 train_data: Optional[List[UserData]] = None,
                 eval_data: Optional[List[UserData]] = None,
                 statement_pool: Optional[List[str]] = None,
                 output_dir: str = "./outputs",
                 use_wandb: bool = True):
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
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        
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
            learnable_temperature=True
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Initialize scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
            eta_min=config.learning_rate * 0.01
        )
        
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
                logging.FileHandler(self.output_dir / "training.log"),
                logging.StreamHandler()
            ]
        )
        
        if self.use_wandb:
            wandb.init(
                project="universal-user-embeddings",
                config=vars(self.config),
                name=f"uue_{self.config.base_model_name.split('/')[-1]}"
            )
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
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
            
            loss = loss_result["loss"]
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_acc += loss_result["avg_acc"].item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{loss_result['avg_acc'].item():.4f}"
            })
            
            # Log to wandb
            if self.use_wandb and batch_idx % self.config.logging_steps == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/accuracy": loss_result["avg_acc"].item(),
                    "train/temperature": loss_result["temperature"].item(),
                    "train/learning_rate": self.optimizer.param_groups[0]["lr"],
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
                    self.save_model("best_model.pt")
            
            # Save checkpoint
            if self.global_step % self.config.save_steps == 0:
                self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")
        
        # Update learning rate
        self.scheduler.step()
        
        return {
            "train/loss": total_loss / num_batches,
            "train/accuracy": total_acc / num_batches
        }
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on validation data.
        
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
        logging.info(f"Evaluation metrics: {metrics}")
        
        if self.use_wandb:
            wandb.log(metrics, step=self.global_step)
    
    def save_model(self, filename: str):
        """Save the model."""
        model_path = self.output_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "global_step": self.global_step,
            "best_eval_acc": self.best_eval_acc
        }, model_path)
        logging.info(f"Model saved to {model_path}")
    
    def save_checkpoint(self, filename: str):
        """Save a training checkpoint."""
        checkpoint_path = self.output_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "global_step": self.global_step,
            "best_eval_acc": self.best_eval_acc,
            "best_eval_loss": self.best_eval_loss
        }, checkpoint_path)
        logging.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load a training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_eval_acc = checkpoint["best_eval_acc"]
        self.best_eval_loss = checkpoint["best_eval_loss"]
        
        logging.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def train(self):
        """Main training loop."""
        logging.info("Starting training...")
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
                self.save_model("best_model.pt")
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")
        
        # Save final model
        self.save_model("final_model.pt")
        logging.info("Training completed!")
        
        if self.use_wandb:
            wandb.finish()
    
    def test_user_embedding(self, user_chunks: List[str]) -> torch.Tensor:
        """
        Test the model by getting an embedding for a user.
        
        Args:
            user_chunks: List of text chunks for the user
            
        Returns:
            User embedding vector
        """
        self.model.eval()
        with torch.no_grad():
            embedding = self.model.get_user_embedding(user_chunks)
        return embedding
    
    def test_statement_embedding(self, statement: str, context: Optional[str] = None) -> torch.Tensor:
        """
        Test the model by getting an embedding for a statement.
        
        Args:
            statement: Behavioral statement
            context: Optional context string
            
        Returns:
            Statement embedding vector
        """
        self.model.eval()
        with torch.no_grad():
            embedding = self.model.get_statement_embedding(statement, context)
        return embedding
    
    def compute_similarity(self, user_chunks: List[str], statement: str, context: Optional[str] = None) -> float:
        """
        Compute similarity between a user and a statement.
        
        Args:
            user_chunks: List of text chunks for the user
            statement: Behavioral statement
            context: Optional context string
            
        Returns:
            Similarity score (cosine similarity)
        """
        user_embedding = self.test_user_embedding(user_chunks)
        statement_embedding = self.test_statement_embedding(statement, context)
        
        similarity = torch.sum(user_embedding * statement_embedding).item()
        return similarity


def main():
    """Main function for training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Universal User Embeddings")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = ModelConfig(**config_dict)
    else:
        config = ModelConfig()  # Use default config
    
    # Create trainer
    trainer = UniversalUserEmbeddingTrainer(
        config=config,
        output_dir=args.output_dir,
        use_wandb=not args.no_wandb
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main() 