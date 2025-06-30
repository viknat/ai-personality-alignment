#!/usr/bin/env python3
"""
Simple Example: Universal User Embeddings

This script demonstrates the basic usage of the Universal User Embeddings system.
It shows how to train a model and use it for inference.
"""

import torch
from pathlib import Path

from src.schemas.universal_schema import ModelConfig, UserData, UserChunk
from src.train.trainer import UniversalUserEmbeddingTrainer
from src.train.evaluation import UniversalUserEmbeddingEvaluator


def create_sample_users():
    """Create sample user data for demonstration."""
    users = [
        UserData(
            user_id="tech_user",
            chunks=[
                UserChunk(text="I love programming in Python and building machine learning models.", platform="reddit"),
                UserChunk(text="Just finished reading a paper on transformer architectures.", platform="twitter"),
                UserChunk(text="Working on a new AI project that uses deep learning.", platform="github")
            ],
            statements=[
                "interested in machine learning",
                "enjoys programming",
                "reads research papers",
                "works on AI projects"
            ]
        ),
        UserData(
            user_id="fitness_user",
            chunks=[
                UserChunk(text="Had an amazing workout at the gym today! Deadlifts are my favorite.", platform="instagram"),
                UserChunk(text="Planning my meal prep for the week. Protein and vegetables are key.", platform="whatsapp"),
                UserChunk(text="Just completed a 10-mile run. Training for my next marathon.", platform="strava")
            ],
            statements=[
                "enjoys working out",
                "goes to the gym regularly",
                "plans meals carefully",
                "runs marathons"
            ]
        ),
        UserData(
            user_id="art_user",
            chunks=[
                UserChunk(text="Visited the modern art museum today. The contemporary exhibits were incredible.", platform="instagram"),
                UserChunk(text="Working on a new painting using acrylics. Abstract expressionism is my style.", platform="pinterest"),
                UserChunk(text="Just finished reading a book about Renaissance artists.", platform="goodreads")
            ],
            statements=[
                "appreciates art",
                "visits museums",
                "paints as a hobby",
                "enjoys reading about art history"
            ]
        )
    ]
    return users


def main():
    """Main example function."""
    print("Universal User Embeddings - Simple Example")
    print("=" * 50)
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create sample data
    print("\n1. Creating sample data...")
    train_users = create_sample_users()
    eval_users = create_sample_users()  # Using same data for simplicity
    
    # Create statement pool
    statement_pool = [
        "likes technology",
        "enjoys sports",
        "loves music",
        "interested in cooking",
        "enjoys traveling",
        "plays video games",
        "interested in politics",
        "loves animals",
        "enjoys gardening",
        "interested in history"
    ]
    
    print(f"   Created {len(train_users)} users")
    print(f"   Statement pool size: {len(statement_pool)}")
    
    # Create configuration
    print("\n2. Setting up configuration...")
    config = ModelConfig(
        base_model_name="Qwen/Qwen3-Embedding-0.6B",
        embedding_dim=1024,
        batch_size=2,  # Small batch size for demo
        learning_rate=2e-5,
        temperature=0.1,
        num_epochs=2,  # Few epochs for demo
        eval_steps=10,
        save_steps=20,
        logging_steps=5,
        num_negative_samples=3
    )
    
    print(f"   Base model: {config.base_model_name}")
    print(f"   Embedding dimension: {config.embedding_dim}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Epochs: {config.num_epochs}")
    
    # Create trainer
    print("\n3. Initializing trainer...")
    trainer = UniversalUserEmbeddingTrainer(
        config=config,
        train_data=train_users,
        eval_data=eval_users,
        statement_pool=statement_pool,
        output_dir="./example_outputs",
        use_wandb=False  # Disable wandb for this example
    )
    
    # Train the model
    print("\n4. Training model...")
    trainer.train()
    
    # Load the trained model for inference
    print("\n5. Loading model for inference...")
    model_path = "./example_outputs/best_model.pt"
    evaluator = UniversalUserEmbeddingEvaluator(model_path)
    
    # Test inference
    print("\n6. Testing inference...")
    
    # Example user chunks
    test_user_chunks = [
        "I'm a software engineer who loves coding in Python.",
        "Working on machine learning projects and reading AI papers.",
        "Building neural networks for computer vision tasks."
    ]
    
    # Test statements
    test_statements = [
        "interested in programming",
        "works in technology",
        "enjoys machine learning",
        "loves sports",  # Should be less similar
        "enjoys cooking"  # Should be less similar
    ]
    
    print("   Testing user-statement similarities:")
    print("   " + "-" * 40)
    
    for statement in test_statements:
        similarity = evaluator.compute_similarity(test_user_chunks, statement)
        print(f"   '{statement}': {similarity:.4f}")
    
    # Test statement similarity
    print("\n   Testing statement similarities:")
    print("   " + "-" * 40)
    
    statement_pairs = [
        ("interested in programming", "works in technology"),
        ("interested in programming", "loves sports"),
        ("enjoys machine learning", "works in technology")
    ]
    
    for stmt1, stmt2 in statement_pairs:
        similarity = evaluator.test_statement_similarity(stmt1, stmt2)
        print(f"   '{stmt1}' <-> '{stmt2}': {similarity:.4f}")
    
    # Get embeddings
    print("\n7. Extracting embeddings...")
    user_embedding = evaluator.get_user_embedding(test_user_chunks)
    statement_embedding = evaluator.get_statement_embedding("interested in programming")
    
    print(f"   User embedding shape: {user_embedding.shape}")
    print(f"   Statement embedding shape: {statement_embedding.shape}")
    print(f"   Embedding norm: {torch.norm(torch.tensor(user_embedding)):.4f}")
    
    print("\n" + "=" * 50)
    print("Example completed successfully!")
    print("Check ./example_outputs/ for model files and logs.")


if __name__ == "__main__":
    main() 