"""
Demo script for Universal User Embeddings with Two-Tower Architecture.

This script demonstrates the complete pipeline:
1. Training a two-tower model with separate user and item towers
2. Evaluating the model performance
3. Demonstrating inference capabilities
4. Showing statement similarity analysis
5. Generating a comprehensive report
"""

import torch
import numpy as np
from typing import List, Dict, Any
import logging
from pathlib import Path

from .trainer import UniversalUserEmbeddingTrainer
from .evaluation import UniversalUserEmbeddingEvaluator
from .data_loader import create_sample_data, create_evaluation_data
from ..schemas.universal_schema import ModelConfig, UserData


def run_two_tower_demo():
    """
    Run the complete two-tower demo pipeline.
    
    This demonstrates:
    - Training with separate user and item towers
    - Evaluation with comprehensive metrics
    - Inference capabilities
    - Similarity analysis
    """
    print("="*60)
    print("UNIVERSAL USER EMBEDDINGS - TWO-TOWER DEMO")
    print("="*60)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create configuration for two-tower model
    config = ModelConfig(
        # Two-tower configuration
        user_model_name="Qwen/Qwen3-Embedding-0.6B",
        item_model_name="Qwen/Qwen3-Embedding-0.6B",
        embedding_dim=1024,
        
        # Training parameters
        batch_size=16,  # Smaller batch for demo
        learning_rate=2e-5,
        temperature=0.1,
        num_epochs=3,  # Fewer epochs for demo
        
        # Data processing
        num_chunks_per_user=50,
        num_negative_samples=10,
        
        # Architecture specific
        use_attention_pooling=True,
        attention_heads=8,
        dropout=0.1,
        
        # Metadata support
        use_user_metadata=True,
        use_item_metadata=True,
        metadata_dim=32,
        item_metadata_dim=32,
        
        # Logging
        eval_steps=100,
        save_steps=200,
        logging_steps=50
    )
    
    print(f"\nConfiguration:")
    print(f"  User Tower: {config.user_model_name}")
    print(f"  Item Tower: {config.item_model_name}")
    print(f"  Embedding Dimension: {config.embedding_dim}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Epochs: {config.num_epochs}")
    
    # Step 1: Create sample data
    print("\n" + "="*40)
    print("STEP 1: CREATING SAMPLE DATA")
    print("="*40)
    
    train_data, statement_pool = create_sample_data(config)
    eval_data, eval_statements = create_evaluation_data(config)
    
    print(f"Created {len(train_data)} training users")
    print(f"Created {len(eval_data)} evaluation users")
    print(f"Statement pool size: {len(statement_pool)}")
    print(f"Evaluation statements: {len(eval_statements)}")
    
    # Step 2: Train the two-tower model
    print("\n" + "="*40)
    print("STEP 2: TRAINING TWO-TOWER MODEL")
    print("="*40)
    
    trainer = UniversalUserEmbeddingTrainer(
        config=config,
        train_data=train_data,
        eval_data=eval_data,
        statement_pool=statement_pool,
        output_dir="./demo_outputs",
        use_wandb=False,  # Disable wandb for demo
        use_mixed_precision=True
    )
    
    # Train the model
    trainer.train()
    
    # Step 3: Evaluate the model
    print("\n" + "="*40)
    print("STEP 3: EVALUATING TWO-TOWER MODEL")
    print("="*40)
    
    model_path = "./demo_outputs/best_two_tower_model.pt"
    evaluator = UniversalUserEmbeddingEvaluator(model_path)
    
    # Generate comprehensive evaluation report
    report = evaluator.generate_evaluation_report(
        test_users=eval_data,
        candidate_statements=statement_pool,
        test_statements=eval_statements
    )
    
    # Step 4: Demonstrate inference capabilities
    print("\n" + "="*40)
    print("STEP 4: INFERENCE DEMONSTRATION")
    print("="*40)
    
    # Test user similarity
    test_user_chunks = [
        "I love programming and working on AI projects",
        "Reading research papers about machine learning",
        "Building neural networks and experimenting with new algorithms"
    ]
    
    test_statements = [
        "interested in technology",
        "enjoys reading research papers",
        "works on AI projects",
        "loves programming",
        "interested in machine learning",
        "enjoys outdoor activities",
        "loves cooking",
        "interested in sports"
    ]
    
    print(f"\nTesting user: {test_user_chunks[0][:50]}...")
    print("\nSimilarity scores with different statements:")
    
    similarities = []
    for statement in test_statements:
        similarity = evaluator.compute_similarity(test_user_chunks, statement)
        similarities.append((statement, similarity))
        print(f"  '{statement}': {similarity:.4f}")
    
    # Find most similar statements
    similarities.sort(key=lambda x: x[1], reverse=True)
    print(f"\nTop 3 most similar statements:")
    for i, (statement, sim) in enumerate(similarities[:3], 1):
        print(f"  {i}. '{statement}': {sim:.4f}")
    
    # Step 5: Statement similarity analysis
    print("\n" + "="*40)
    print("STEP 5: STATEMENT SIMILARITY ANALYSIS")
    print("="*40)
    
    tech_statements = [
        "interested in technology",
        "loves programming",
        "works on AI projects",
        "interested in machine learning"
    ]
    
    non_tech_statements = [
        "enjoys outdoor activities",
        "loves cooking",
        "interested in sports",
        "enjoys reading fiction"
    ]
    
    print("\nTech statement similarities:")
    for i, stmt1 in enumerate(tech_statements):
        for j, stmt2 in enumerate(tech_statements[i+1:], i+1):
            sim = evaluator.test_statement_similarity(stmt1, stmt2)
            print(f"  '{stmt1}' vs '{stmt2}': {sim:.4f}")
    
    print("\nNon-tech statement similarities:")
    for i, stmt1 in enumerate(non_tech_statements):
        for j, stmt2 in enumerate(non_tech_statements[i+1:], i+1):
            sim = evaluator.test_statement_similarity(stmt1, stmt2)
            print(f"  '{stmt1}' vs '{stmt2}': {sim:.4f}")
    
    print("\nCross-category similarities:")
    for tech_stmt in tech_statements[:2]:
        for non_tech_stmt in non_tech_statements[:2]:
            sim = evaluator.test_statement_similarity(tech_stmt, non_tech_stmt)
            print(f"  '{tech_stmt}' vs '{non_tech_stmt}': {sim:.4f}")
    
    # Step 6: User embedding analysis
    print("\n" + "="*40)
    print("STEP 6: USER EMBEDDING ANALYSIS")
    print("="*40)
    
    # Get embeddings for different types of users
    tech_user_chunks = [
        "I love programming and working on AI projects",
        "Reading research papers about machine learning",
        "Building neural networks and experimenting with new algorithms"
    ]
    
    cooking_user_chunks = [
        "I love cooking and trying new recipes",
        "Experimenting with different cuisines",
        "Reading cookbooks and food blogs"
    ]
    
    sports_user_chunks = [
        "I love playing basketball and watching sports",
        "Going to the gym regularly",
        "Following professional sports teams"
    ]
    
    # Get embeddings
    tech_embedding = evaluator.get_user_embedding(tech_user_chunks)
    cooking_embedding = evaluator.get_user_embedding(cooking_user_chunks)
    sports_embedding = evaluator.get_user_embedding(sports_user_chunks)
    
    # Compute similarities between users
    tech_cooking_sim = np.dot(tech_embedding, cooking_embedding)
    tech_sports_sim = np.dot(tech_embedding, sports_embedding)
    cooking_sports_sim = np.dot(cooking_embedding, sports_embedding)
    
    print(f"\nUser similarity analysis:")
    print(f"  Tech user vs Cooking user: {tech_cooking_sim:.4f}")
    print(f"  Tech user vs Sports user: {tech_sports_sim:.4f}")
    print(f"  Cooking user vs Sports user: {cooking_sports_sim:.4f}")
    
    # Step 7: Generate final report
    print("\n" + "="*40)
    print("STEP 7: FINAL DEMO REPORT")
    print("="*40)
    
    # Save demo results
    demo_results = {
        "model_config": {
            "user_model": config.user_model_name,
            "item_model": config.item_model_name,
            "embedding_dim": config.embedding_dim,
            "temperature": config.temperature
        },
        "training": {
            "num_train_users": len(train_data),
            "num_eval_users": len(eval_data),
            "statement_pool_size": len(statement_pool),
            "best_eval_accuracy": trainer.best_eval_acc
        },
        "evaluation": report,
        "inference_demo": {
            "test_user": test_user_chunks[0],
            "top_similar_statements": similarities[:3]
        },
        "user_similarities": {
            "tech_cooking": float(tech_cooking_sim),
            "tech_sports": float(tech_sports_sim),
            "cooking_sports": float(cooking_sports_sim)
        }
    }
    
    # Save results
    import json
    output_path = Path("./demo_outputs/demo_results.json")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(demo_results, f, indent=2)
    
    print(f"\nDemo results saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("DEMO SUMMARY")
    print("="*60)
    print(f"✅ Two-tower model trained successfully")
    print(f"✅ Model evaluation completed")
    print(f"✅ Inference capabilities demonstrated")
    print(f"✅ Similarity analysis performed")
    print(f"✅ User embedding analysis completed")
    print(f"✅ Results saved to ./demo_outputs/")
    
    print(f"\nKey Results:")
    print(f"  Best Evaluation Accuracy: {trainer.best_eval_acc:.4f}")
    print(f"  Preference Retrieval F1@10: {report['preference_retrieval']['f1@k']:.4f}")
    print(f"  Zero-shot Inference F1: {report['zero_shot_inference']['f1']:.4f}")
    print(f"  Zero-shot Inference AUC: {report['zero_shot_inference']['auc']:.4f}")
    
    print(f"\nThe two-tower model successfully learned to:")
    print(f"  • Encode users and items into a shared embedding space")
    print(f"  • Distinguish between related and unrelated statements")
    print(f"  • Capture semantic similarities between different user types")
    print(f"  • Perform zero-shot preference inference")
    
    return demo_results


def main():
    """Main function to run the demo."""
    try:
        results = run_two_tower_demo()
        print(f"\n🎉 Demo completed successfully!")
        return results
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main() 