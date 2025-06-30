"""
Evaluation Module for Universal User Embeddings.

This module provides comprehensive evaluation capabilities for the trained model,
including preference retrieval, zero-shot inference, and personality trait analysis.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import json
from pathlib import Path

from .model import UniversalUserEmbeddingModel
from .trainer import UniversalUserEmbeddingTrainer
from ..schemas.universal_schema import ModelConfig, UserData


class UniversalUserEmbeddingEvaluator:
    """
    Evaluator for Universal User Embeddings model.
    
    This class provides comprehensive evaluation capabilities:
    - Preference statement retrieval
    - Zero-shot preference inference
    - Personality trait regression
    - User similarity analysis
    """
    
    def __init__(self, 
                 model_path: str,
                 config: Optional[ModelConfig] = None,
                 device: Optional[str] = None):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to the trained model checkpoint
            config: Model configuration (optional, loaded from checkpoint if not provided)
            device: Device to run evaluation on
        """
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if config is None:
            config = checkpoint["config"]
        
        self.config = config
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Initialize model
        self.model = UniversalUserEmbeddingModel(config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Using device: {self.device}")
    
    def evaluate_preference_retrieval(self, 
                                    test_users: List[UserData],
                                    candidate_statements: List[str],
                                    top_k: int = 10) -> Dict[str, float]:
        """
        Evaluate preference statement retrieval performance.
        
        Args:
            test_users: List of test users with known preferences
            candidate_statements: Pool of candidate statements to retrieve from
            top_k: Number of top statements to retrieve
            
        Returns:
            Dictionary of retrieval metrics
        """
        print("Evaluating preference statement retrieval...")
        
        all_precisions = []
        all_recalls = []
        all_f1_scores = []
        
        for user in test_users:
            # Get user embedding
            user_chunks = [chunk.text for chunk in user.chunks]
            user_embedding = self.model.get_user_embedding(user_chunks)
            
            # Get statement embeddings
            statement_embeddings = []
            for statement in candidate_statements:
                stmt_embedding = self.model.get_statement_embedding(statement)
                statement_embeddings.append(stmt_embedding)
            
            statement_embeddings = torch.stack(statement_embeddings)
            
            # Compute similarities
            similarities = torch.matmul(user_embedding.unsqueeze(0), statement_embeddings.t()).squeeze(0)
            
            # Get top-k predictions
            top_k_indices = torch.topk(similarities, k=min(top_k, len(candidate_statements)))[1]
            predicted_statements = [candidate_statements[i] for i in top_k_indices]
            
            # Create ground truth labels
            ground_truth = [1 if stmt in user.statements else 0 for stmt in candidate_statements]
            predicted = [1 if stmt in predicted_statements else 0 for stmt in candidate_statements]
            
            # Compute metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                ground_truth, predicted, average='binary', zero_division=0
            )
            
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1_scores.append(f1)
        
        return {
            "precision@k": np.mean(all_precisions),
            "recall@k": np.mean(all_recalls),
            "f1@k": np.mean(all_f1_scores),
            "std_precision": np.std(all_precisions),
            "std_recall": np.std(all_recalls),
            "std_f1": np.std(all_f1_scores)
        }
    
    def evaluate_zero_shot_inference(self,
                                   test_users: List[UserData],
                                   test_statements: List[str]) -> Dict[str, float]:
        """
        Evaluate zero-shot preference inference.
        
        Args:
            test_users: List of test users
            test_statements: List of statements to test
            
        Returns:
            Dictionary of zero-shot inference metrics
        """
        print("Evaluating zero-shot preference inference...")
        
        all_similarities = []
        all_labels = []
        
        for user in test_users:
            # Get user embedding
            user_chunks = [chunk.text for chunk in user.chunks]
            user_embedding = self.model.get_user_embedding(user_chunks)
            
            for statement in test_statements:
                # Get statement embedding
                stmt_embedding = self.model.get_statement_embedding(statement)
                
                # Compute similarity
                similarity = torch.sum(user_embedding * stmt_embedding).item()
                all_similarities.append(similarity)
                
                # Create label (1 if statement is in user's statements, 0 otherwise)
                label = 1 if statement in user.statements else 0
                all_labels.append(label)
        
        # Convert to numpy arrays
        similarities = np.array(all_similarities)
        labels = np.array(all_labels)
        
        # Find optimal threshold
        thresholds = np.linspace(-1, 1, 100)
        best_f1 = 0
        best_threshold = 0
        
        for threshold in thresholds:
            predictions = (similarities > threshold).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        # Compute final metrics with best threshold
        final_predictions = (similarities > best_threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, final_predictions, average='binary', zero_division=0
        )
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "best_threshold": best_threshold,
            "auc": np.trapz(recall, precision) if len(np.unique(labels)) > 1 else 0.5
        }
    
    def evaluate_personality_traits(self,
                                  users_with_traits: List[Tuple[UserData, Dict[str, float]]],
                                  trait_names: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate personality trait regression performance.
        
        Args:
            users_with_traits: List of (user, traits_dict) tuples
            trait_names: List of trait names to evaluate
            
        Returns:
            Dictionary of regression metrics for each trait
        """
        print("Evaluating personality trait regression...")
        
        results = {}
        
        for trait in trait_names:
            # Extract embeddings and trait values
            embeddings = []
            trait_values = []
            
            for user, traits in users_with_traits:
                if trait in traits:
                    user_chunks = [chunk.text for chunk in user.chunks]
                    user_embedding = self.model.get_user_embedding(user_chunks)
                    embeddings.append(user_embedding.cpu().numpy())
                    trait_values.append(traits[trait])
            
            if len(embeddings) < 10:  # Need sufficient data
                print(f"Warning: Insufficient data for trait {trait}")
                continue
            
            embeddings = np.array(embeddings)
            trait_values = np.array(trait_values)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                embeddings, trait_values, test_size=0.3, random_state=42
            )
            
            # Train regression model
            regressor = LogisticRegression(random_state=42, max_iter=1000)
            regressor.fit(X_train, y_train)
            
            # Evaluate
            y_pred = regressor.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Compute correlation
            correlation = np.corrcoef(y_test, y_pred)[0, 1] if len(y_test) > 1 else 0
            
            results[trait] = {
                "accuracy": accuracy,
                "correlation": correlation,
                "n_samples": len(embeddings)
            }
        
        return results
    
    def analyze_user_similarities(self, users: List[UserData]) -> Dict[str, Any]:
        """
        Analyze user similarities in the embedding space.
        
        Args:
            users: List of users to analyze
            
        Returns:
            Dictionary containing similarity analysis
        """
        print("Analyzing user similarities...")
        
        # Get user embeddings
        user_embeddings = []
        user_ids = []
        
        for user in users:
            user_chunks = [chunk.text for chunk in user.chunks]
            embedding = self.model.get_user_embedding(user_chunks)
            user_embeddings.append(embedding.cpu().numpy())
            user_ids.append(user.user_id)
        
        user_embeddings = np.array(user_embeddings)
        
        # Compute pairwise similarities
        similarity_matrix = np.matmul(user_embeddings, user_embeddings.T)
        
        # Find most similar pairs
        n_users = len(users)
        most_similar_pairs = []
        
        for i in range(n_users):
            for j in range(i + 1, n_users):
                similarity = similarity_matrix[i, j]
                most_similar_pairs.append((user_ids[i], user_ids[j], similarity))
        
        # Sort by similarity
        most_similar_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Compute clustering statistics
        mean_similarity = np.mean(similarity_matrix[np.triu_indices(n_users, k=1)])
        std_similarity = np.std(similarity_matrix[np.triu_indices(n_users, k=1)])
        
        return {
            "similarity_matrix": similarity_matrix.tolist(),
            "most_similar_pairs": most_similar_pairs[:10],  # Top 10
            "mean_similarity": mean_similarity,
            "std_similarity": std_similarity,
            "user_ids": user_ids
        }
    
    def test_statement_similarity(self, 
                                statement1: str, 
                                statement2: str,
                                context1: Optional[str] = None,
                                context2: Optional[str] = None) -> float:
        """
        Test similarity between two statements.
        
        Args:
            statement1: First statement
            statement2: Second statement
            context1: Optional context for first statement
            context2: Optional context for second statement
            
        Returns:
            Similarity score
        """
        embedding1 = self.model.get_statement_embedding(statement1, context1)
        embedding2 = self.model.get_statement_embedding(statement2, context2)
        
        similarity = torch.sum(embedding1 * embedding2).item()
        return similarity
    
    def get_user_embedding(self, user_chunks: List[str]) -> np.ndarray:
        """
        Get embedding for a user.
        
        Args:
            user_chunks: List of text chunks for the user
            
        Returns:
            User embedding as numpy array
        """
        embedding = self.model.get_user_embedding(user_chunks)
        return embedding.cpu().numpy()
    
    def get_statement_embedding(self, statement: str, context: Optional[str] = None) -> np.ndarray:
        """
        Get embedding for a statement.
        
        Args:
            statement: Behavioral statement
            context: Optional context string
            
        Returns:
            Statement embedding as numpy array
        """
        embedding = self.model.get_statement_embedding(statement, context)
        return embedding.cpu().numpy()


def create_evaluation_report(evaluator: UniversalUserEmbeddingEvaluator,
                           test_users: List[UserData],
                           candidate_statements: List[str],
                           output_path: str = "evaluation_report.json"):
    """
    Create a comprehensive evaluation report.
    
    Args:
        evaluator: Initialized evaluator
        test_users: Test users for evaluation
        candidate_statements: Candidate statements for retrieval
        output_path: Path to save the report
    """
    print("Creating comprehensive evaluation report...")
    
    report = {
        "model_info": {
            "base_model": evaluator.config.base_model_name,
            "embedding_dim": evaluator.config.embedding_dim,
            "temperature": evaluator.config.temperature
        },
        "evaluation_results": {}
    }
    
    # Preference retrieval evaluation
    retrieval_metrics = evaluator.evaluate_preference_retrieval(
        test_users, candidate_statements, top_k=10
    )
    report["evaluation_results"]["preference_retrieval"] = retrieval_metrics
    
    # Zero-shot inference evaluation
    zero_shot_metrics = evaluator.evaluate_zero_shot_inference(
        test_users, candidate_statements
    )
    report["evaluation_results"]["zero_shot_inference"] = zero_shot_metrics
    
    # User similarity analysis
    similarity_analysis = evaluator.analyze_user_similarities(test_users)
    report["evaluation_results"]["user_similarities"] = similarity_analysis
    
    # Save report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Evaluation report saved to {output_path}")
    
    # Print summary
    print("\n=== EVALUATION SUMMARY ===")
    print(f"Preference Retrieval F1@10: {retrieval_metrics['f1@k']:.4f}")
    print(f"Zero-shot Inference F1: {zero_shot_metrics['f1_score']:.4f}")
    print(f"Mean User Similarity: {similarity_analysis['mean_similarity']:.4f}")
    
    return report


def main():
    """Main function for evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Universal User Embeddings")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--test_data", type=str, help="Path to test data JSON")
    parser.add_argument("--output_dir", type=str, default="./evaluation", help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize evaluator
    evaluator = UniversalUserEmbeddingEvaluator(args.model_path)
    
    # Load test data if provided
    if args.test_data:
        with open(args.test_data, 'r') as f:
            test_data = json.load(f)
        # Convert to UserData objects
        test_users = []
        for user_data in test_data["users"]:
            # Implementation depends on your data format
            pass
    else:
        # Use sample data for demonstration
        from .data_loader import create_evaluation_data
        test_users, candidate_statements = create_evaluation_data(evaluator.config)
    
    # Create evaluation report
    report = create_evaluation_report(
        evaluator, test_users, candidate_statements,
        output_path=output_dir / "evaluation_report.json"
    )


if __name__ == "__main__":
    main() 