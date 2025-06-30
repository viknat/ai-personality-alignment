"""
Demo Script for Universal User Embeddings.

This script demonstrates the complete training and evaluation pipeline
for the Universal User Embeddings model.
"""

import torch
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Any

from .trainer import UniversalUserEmbeddingTrainer
from .evaluation import UniversalUserEmbeddingEvaluator, create_evaluation_report
from .data_loader import create_sample_data, create_evaluation_data
from ..schemas.universal_schema import ModelConfig, UserData


def run_training_demo():
    """Run a complete training demo."""
    print("=" * 60)
    print("UNIVERSAL USER EMBEDDINGS - TRAINING DEMO")
    print("=" * 60)
    
    # Create configuration
    config = ModelConfig(
        base_model_name="intfloat/e5-base-v2",
        embedding_dim=768,
        batch_size=8,  # Smaller batch size for demo
        learning_rate=2e-5,
        temperature=0.1,
        num_epochs=3,  # Few epochs for demo
        eval_steps=50,
        save_steps=100,
        logging_steps=10,
        num_negative_samples=5
    )
    
    print(f"Configuration:")
    print(f"  Base Model: {config.base_model_name}")
    print(f"  Embedding Dimension: {config.embedding_dim}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Epochs: {config.num_epochs}")
    
    # Create trainer
    trainer = UniversalUserEmbeddingTrainer(
        config=config,
        output_dir="./demo_outputs",
        use_wandb=False  # Disable wandb for demo
    )
    
    print(f"\nTraining on {len(trainer.train_data)} users")
    print(f"Evaluating on {len(trainer.eval_data)} users")
    print(f"Statement pool size: {len(trainer.statement_pool)}")
    
    # Start training
    print("\nStarting training...")
    trainer.train()
    
    print("\nTraining completed!")
    return trainer


def run_evaluation_demo(trainer: UniversalUserEmbeddingTrainer):
    """Run evaluation demo on the trained model."""
    print("\n" + "=" * 60)
    print("UNIVERSAL USER EMBEDDINGS - EVALUATION DEMO")
    print("=" * 60)
    
    # Create evaluator
    model_path = "./demo_outputs/best_model.pt"
    evaluator = UniversalUserEmbeddingEvaluator(model_path)
    
    # Get test data
    test_users, candidate_statements = create_evaluation_data(trainer.config)
    
    print(f"Evaluating on {len(test_users)} test users")
    print(f"Testing with {len(candidate_statements)} candidate statements")
    
    # Run evaluations
    print("\n1. Preference Retrieval Evaluation...")
    retrieval_metrics = evaluator.evaluate_preference_retrieval(
        test_users, candidate_statements, top_k=5
    )
    print(f"   Precision@5: {retrieval_metrics['precision@k']:.4f}")
    print(f"   Recall@5: {retrieval_metrics['recall@k']:.4f}")
    print(f"   F1@5: {retrieval_metrics['f1@k']:.4f}")
    
    print("\n2. Zero-shot Inference Evaluation...")
    zero_shot_metrics = evaluator.evaluate_zero_shot_inference(
        test_users, candidate_statements
    )
    print(f"   Precision: {zero_shot_metrics['precision']:.4f}")
    print(f"   Recall: {zero_shot_metrics['recall']:.4f}")
    print(f"   F1 Score: {zero_shot_metrics['f1_score']:.4f}")
    print(f"   Best Threshold: {zero_shot_metrics['best_threshold']:.4f}")
    
    print("\n3. User Similarity Analysis...")
    similarity_analysis = evaluator.analyze_user_similarities(test_users)
    print(f"   Mean Similarity: {similarity_analysis['mean_similarity']:.4f}")
    print(f"   Std Similarity: {similarity_analysis['std_similarity']:.4f}")
    
    # Show most similar user pairs
    print("\n   Most Similar User Pairs:")
    for i, (user1, user2, similarity) in enumerate(similarity_analysis['most_similar_pairs'][:3]):
        print(f"     {i+1}. {user1} <-> {user2}: {similarity:.4f}")
    
    return evaluator


def run_inference_demo(evaluator: UniversalUserEmbeddingEvaluator):
    """Run inference demo with example users and statements."""
    print("\n" + "=" * 60)
    print("UNIVERSAL USER EMBEDDINGS - INFERENCE DEMO")
    print("=" * 60)
    
    # Example users
    example_users = [
        {
            "name": "Tech Enthusiast",
            "chunks": [
                "I love programming in Python and building machine learning models.",
                "Just finished reading a paper on transformer architectures.",
                "Working on a new AI project that uses deep learning for image recognition."
            ]
        },
        {
            "name": "Fitness Enthusiast", 
            "chunks": [
                "Had an amazing workout at the gym today! Deadlifts are my favorite.",
                "Planning my meal prep for the week. Protein and vegetables are key.",
                "Just completed a 10-mile run. Training for my next marathon."
            ]
        },
        {
            "name": "Art Lover",
            "chunks": [
                "Visited the modern art museum today. The contemporary exhibits were incredible.",
                "Working on a new painting using acrylics. Abstract expressionism is my style.",
                "Just finished reading a book about Renaissance artists. Da Vinci was truly ahead of his time."
            ]
        }
    ]
    
    # Example statements to test
    test_statements = [
        "interested in technology and programming",
        "enjoys physical exercise and fitness",
        "appreciates art and creativity",
        "loves reading books",
        "enjoys outdoor activities",
        "works in software development",
        "goes to the gym regularly",
        "paints or draws as a hobby"
    ]
    
    print("Testing user-statement similarities:")
    print("-" * 50)
    
    for user_info in example_users:
        print(f"\n{user_info['name']}:")
        user_chunks = user_info["chunks"]
        
        # Get user embedding
        user_embedding = evaluator.get_user_embedding(user_chunks)
        
        # Test each statement
        similarities = []
        for statement in test_statements:
            stmt_embedding = evaluator.get_statement_embedding(statement)
            similarity = np.dot(user_embedding, stmt_embedding)
            similarities.append((statement, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Show top 3 most similar statements
        print("  Top 3 most similar statements:")
        for i, (statement, sim) in enumerate(similarities[:3]):
            print(f"    {i+1}. {statement}: {sim:.4f}")
        
        # Show bottom 3 least similar statements
        print("  Bottom 3 least similar statements:")
        for i, (statement, sim) in enumerate(similarities[-3:]):
            print(f"    {i+1}. {statement}: {sim:.4f}")


def run_statement_similarity_demo(evaluator: UniversalUserEmbeddingEvaluator):
    """Demo statement-to-statement similarities."""
    print("\n" + "=" * 60)
    print("UNIVERSAL USER EMBEDDINGS - STATEMENT SIMILARITY DEMO")
    print("=" * 60)
    
    # Example statement pairs
    statement_pairs = [
        ("loves programming", "enjoys coding"),
        ("goes to the gym", "exercises regularly"),
        ("loves programming", "goes to the gym"),  # Should be less similar
        ("enjoys reading", "loves books"),
        ("plays sports", "enjoys outdoor activities"),
        ("works in tech", "programs software"),
        ("loves cooking", "enjoys gardening"),  # Should be less similar
        ("travels frequently", "loves exploring new places")
    ]
    
    print("Statement Similarity Analysis:")
    print("-" * 50)
    
    for stmt1, stmt2 in statement_pairs:
        similarity = evaluator.test_statement_similarity(stmt1, stmt2)
        print(f"'{stmt1}' <-> '{stmt2}': {similarity:.4f}")


def create_comprehensive_report(trainer: UniversalUserEmbeddingTrainer, 
                              evaluator: UniversalUserEmbeddingEvaluator):
    """Create a comprehensive demo report."""
    print("\n" + "=" * 60)
    print("CREATING COMPREHENSIVE DEMO REPORT")
    print("=" * 60)
    
    # Get test data
    test_users, candidate_statements = create_evaluation_data(trainer.config)
    
    # Create evaluation report
    report_path = "./demo_outputs/comprehensive_report.json"
    report = create_evaluation_report(
        evaluator, test_users, candidate_statements, report_path
    )
    
    # Add demo-specific information
    demo_info = {
        "demo_configuration": {
            "base_model": trainer.config.base_model_name,
            "embedding_dim": trainer.config.embedding_dim,
            "training_epochs": trainer.config.num_epochs,
            "batch_size": trainer.config.batch_size,
            "learning_rate": trainer.config.learning_rate
        },
        "training_summary": {
            "num_train_users": len(trainer.train_data),
            "num_eval_users": len(trainer.eval_data),
            "statement_pool_size": len(trainer.statement_pool),
            "best_eval_accuracy": trainer.best_eval_acc
        }
    }
    
    report["demo_info"] = demo_info
    
    # Save updated report
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Comprehensive report saved to {report_path}")
    
    return report


def main():
    """Main demo function."""
    print("Universal User Embeddings - Complete Demo")
    print("This demo will:")
    print("1. Train a Universal User Embeddings model")
    print("2. Evaluate the model performance")
    print("3. Demonstrate inference capabilities")
    print("4. Show statement similarity analysis")
    print("5. Generate a comprehensive report")
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"\nCUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("\nUsing CPU for training")
    
    try:
        # Step 1: Training
        trainer = run_training_demo()
        
        # Step 2: Evaluation
        evaluator = run_evaluation_demo(trainer)
        
        # Step 3: Inference Demo
        run_inference_demo(evaluator)
        
        # Step 4: Statement Similarity Demo
        run_statement_similarity_demo(evaluator)
        
        # Step 5: Comprehensive Report
        report = create_comprehensive_report(trainer, evaluator)
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Check the ./demo_outputs/ directory for:")
        print("- Trained model checkpoints")
        print("- Evaluation reports")
        print("- Training logs")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 