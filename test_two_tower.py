#!/usr/bin/env python3
"""
Test script for the two-tower Universal User Embeddings implementation.

This script verifies that:
1. The two-tower model can be instantiated correctly
2. The model can process user and item inputs
3. The training pipeline works
4. The evaluation pipeline works
"""

import torch
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from src.schemas.universal_schema import ModelConfig
from src.train.model import UniversalUserEmbeddingModel
from src.train.trainer import UniversalUserEmbeddingTrainer
from src.train.evaluation import UniversalUserEmbeddingEvaluator
from src.train.data_loader import create_sample_data, create_evaluation_data


def test_model_instantiation():
    """Test that the two-tower model can be instantiated."""
    print("Testing model instantiation...")
    
    config = ModelConfig(
        user_model_name="Qwen/Qwen3-Embedding-0.6B",
        item_model_name="Qwen/Qwen3-Embedding-0.6B",
        embedding_dim=1024,
        batch_size=4,
        num_epochs=1
    )
    
    model = UniversalUserEmbeddingModel(config)
    print(f"✅ Model instantiated successfully")
    print(f"   User tower parameters: {sum(p.numel() for p in model.user_tower.parameters()):,}")
    print(f"   Item tower parameters: {sum(p.numel() for p in model.item_tower.parameters()):,}")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, config


def test_model_forward_pass(model, config):
    """Test that the model can process inputs correctly."""
    print("\nTesting forward pass...")
    
    # Create sample inputs
    user_chunks = [
        ["I love programming", "Working on AI projects"],
        ["I enjoy reading books", "Love cooking"]
    ]
    
    positive_statements = [
        "interested in technology",
        "enjoys reading"
    ]
    
    negative_statements = [
        ["loves sports", "enjoys outdoor activities"],
        ["works in finance", "loves shopping"]
    ]
    
    # Forward pass
    outputs = model(
        user_chunks=user_chunks,
        positive_statements=positive_statements,
        negative_statements=negative_statements
    )
    
    print(f"✅ Forward pass successful")
    print(f"   User embeddings shape: {outputs['user_embeddings'].shape}")
    print(f"   Positive embeddings shape: {outputs['positive_embeddings'].shape}")
    print(f"   Negative embeddings shape: {outputs['negative_embeddings'].shape}")
    print(f"   Temperature: {outputs['temperature'].item():.4f}")
    
    return outputs


def test_training_pipeline(config):
    """Test the training pipeline."""
    print("\nTesting training pipeline...")
    
    # Create sample data
    train_data, statement_pool = create_sample_data(config)
    eval_data, eval_statements = create_evaluation_data(config)
    
    print(f"✅ Sample data created")
    print(f"   Training users: {len(train_data)}")
    print(f"   Evaluation users: {len(eval_data)}")
    print(f"   Statement pool: {len(statement_pool)}")
    
    # Create trainer
    trainer = UniversalUserEmbeddingTrainer(
        config=config,
        train_data=train_data,
        eval_data=eval_data,
        statement_pool=statement_pool,
        output_dir="./test_outputs",
        use_wandb=False,
        use_mixed_precision=False  # Disable for testing
    )
    
    print(f"✅ Trainer created successfully")
    print(f"   User optimizer: {type(trainer.user_optimizer).__name__}")
    print(f"   Item optimizer: {type(trainer.item_optimizer).__name__}")
    
    # Train for one epoch
    print("\nTraining for one epoch...")
    train_metrics = trainer.train_epoch(0)
    
    print(f"✅ Training completed")
    print(f"   Loss: {train_metrics['train/loss']:.4f}")
    print(f"   Accuracy: {train_metrics['train/accuracy']:.4f}")
    
    return trainer


def test_evaluation_pipeline(trainer):
    """Test the evaluation pipeline."""
    print("\nTesting evaluation pipeline...")
    
    # Save a checkpoint
    trainer.save_model("test_model.pt")
    
    # Create evaluator with the correct path
    model_path = "./test_outputs/test_model.pt"
    evaluator = UniversalUserEmbeddingEvaluator(model_path)
    
    print(f"✅ Evaluator created successfully")
    
    # Test basic functionality
    test_user_chunks = ["I love programming", "Working on AI projects"]
    test_statement = "interested in technology"
    
    user_embedding = evaluator.get_user_embedding(test_user_chunks)
    statement_embedding = evaluator.get_statement_embedding(test_statement)
    similarity = evaluator.compute_similarity(test_user_chunks, test_statement)
    
    print(f"✅ Basic inference works")
    print(f"   User embedding shape: {user_embedding.shape}")
    print(f"   Statement embedding shape: {statement_embedding.shape}")
    print(f"   Similarity: {similarity:.4f}")
    
    return evaluator


def test_two_tower_architecture():
    """Test the complete two-tower architecture."""
    print("="*60)
    print("TESTING TWO-TOWER ARCHITECTURE")
    print("="*60)
    
    try:
        # Test 1: Model instantiation
        model, config = test_model_instantiation()
        
        # Test 2: Forward pass
        outputs = test_model_forward_pass(model, config)
        
        # Test 3: Training pipeline
        trainer = test_training_pipeline(config)
        
        # Test 4: Evaluation pipeline
        evaluator = test_evaluation_pipeline(trainer)
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("The two-tower architecture is working correctly:")
        print("  • Separate user and item towers")
        print("  • Independent optimizers")
        print("  • Proper forward pass")
        print("  • Training pipeline")
        print("  • Evaluation pipeline")
        print("  • Inference capabilities")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_two_tower_architecture()
    sys.exit(0 if success else 1) 