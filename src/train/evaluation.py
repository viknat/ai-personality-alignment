"""
Evaluation module for Universal User Embeddings with two-tower architecture.

This module provides comprehensive evaluation capabilities for the two-tower model:
- Preference statement retrieval evaluation
- Zero-shot preference inference
- Statement similarity analysis
- User embedding analysis
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
import logging

from .model import UniversalUserEmbeddingModel
from ..schemas.universal_schema import UserData, ModelConfig


class UniversalUserEmbeddingEvaluator:
    """
    Evaluator for Universal User Embeddings with two-tower architecture.
    
    This class provides comprehensive evaluation capabilities for the two-tower model,
    including preference retrieval, zero-shot inference, and similarity analysis.
    """
    
    def __init__(self, model_path: str, config: Optional[ModelConfig] = None):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to the trained model checkpoint
            config: Model configuration (optional, will be loaded from checkpoint)
        """
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if config is None:
            config = checkpoint['config']
        
        self.config = config
        self.model = UniversalUserEmbeddingModel(config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        logging.info(f"Model loaded from {model_path}")
        logging.info(f"Using device: {self.device}")
    
    def evaluate_preference_retrieval(self, 
                                    test_users: List[UserData],
                                    candidate_statements: List[str],
                                    top_k: int = 10) -> Dict[str, float]:
        """
        Evaluate preference statement retrieval performance.
        
        Args:
            test_users: List of test users
            candidate_statements: List of candidate statements to rank
            top_k: Number of top statements to consider
            
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
        
        # Compute final metrics with optimal threshold
        final_predictions = (similarities > best_threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, final_predictions, average='binary', zero_division=0
        )
        
        # Compute AUC and AP
        try:
            auc = roc_auc_score(labels, similarities)
        except ValueError:
            auc = 0.5  # Default if only one class
        
        try:
            ap = average_precision_score(labels, similarities)
        except ValueError:
            ap = 0.0  # Default if only one class
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
            "average_precision": ap,
            "optimal_threshold": best_threshold,
            "num_samples": len(similarities)
        }
    
    def evaluate_user_similarity(self, 
                               test_users: List[UserData],
                               similarity_threshold: float = 0.5) -> Dict[str, float]:
        """
        Evaluate user similarity computation.
        
        Args:
            test_users: List of test users
            similarity_threshold: Threshold for considering users similar
            
        Returns:
            Dictionary of user similarity metrics
        """
        print("Evaluating user similarity computation...")
        
        # Get all user embeddings
        user_embeddings = []
        user_ids = []
        
        for user in test_users:
            user_chunks = [chunk.text for chunk in user.chunks]
            embedding = self.model.get_user_embedding(user_chunks)
            user_embeddings.append(embedding.cpu().numpy())
            user_ids.append(user.user_id)
        
        user_embeddings = np.array(user_embeddings)
        
        # Compute pairwise similarities
        similarity_matrix = cosine_similarity(user_embeddings)
        
        # Analyze similarity distribution
        similarities = []
        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix)):
                similarities.append(similarity_matrix[i, j])
        
        similarities = np.array(similarities)
        
        return {
            "mean_similarity": np.mean(similarities),
            "std_similarity": np.std(similarities),
            "min_similarity": np.min(similarities),
            "max_similarity": np.max(similarities),
            "similarity_above_threshold": np.mean(similarities > similarity_threshold),
            "num_user_pairs": len(similarities)
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
        embedding1 = self.model.get_statement_embedding(statement1, context=context1)
        embedding2 = self.model.get_statement_embedding(statement2, context=context2)
        
        similarity = torch.sum(embedding1 * embedding2).item()
        return similarity
    
    def get_user_embedding(self, user_chunks: List[str], user_metadata: Optional[torch.Tensor] = None) -> np.ndarray:
        """
        Get embedding for a user.
        
        Args:
            user_chunks: List of text chunks for the user
            user_metadata: Optional user metadata
            
        Returns:
            User embedding as numpy array
        """
        embedding = self.model.get_user_embedding(user_chunks, user_metadata)
        return embedding.cpu().numpy()
    
    def get_statement_embedding(self, statement: str, item_metadata: Optional[torch.Tensor] = None, context: Optional[str] = None) -> np.ndarray:
        """
        Get embedding for a statement.
        
        Args:
            statement: Behavioral statement
            item_metadata: Optional item metadata
            context: Optional context string
            
        Returns:
            Statement embedding as numpy array
        """
        embedding = self.model.get_statement_embedding(statement, item_metadata, context)
        return embedding.cpu().numpy()
    
    def compute_similarity(self, user_chunks: List[str], statement: str,
                          user_metadata: Optional[torch.Tensor] = None,
                          item_metadata: Optional[torch.Tensor] = None,
                          context: Optional[str] = None) -> float:
        """
        Compute similarity between a user and a statement.
        
        Args:
            user_chunks: List of text chunks for the user
            statement: Behavioral statement
            user_metadata: Optional user metadata
            item_metadata: Optional item metadata
            context: Optional context string
            
        Returns:
            Similarity score (cosine similarity)
        """
        user_embedding = self.model.get_user_embedding(user_chunks, user_metadata)
        statement_embedding = self.model.get_statement_embedding(statement, item_metadata, context)
        
        similarity = torch.sum(user_embedding * statement_embedding).item()
        return similarity
    
    def analyze_embedding_space(self, test_users: List[UserData], test_statements: List[str]) -> Dict[str, float]:
        """
        Analyze the embedding space properties.
        
        Args:
            test_users: List of test users
            test_statements: List of test statements
            
        Returns:
            Dictionary of embedding space analysis metrics
        """
        print("Analyzing embedding space...")
        
        # Get user embeddings
        user_embeddings = []
        for user in test_users:
            user_chunks = [chunk.text for chunk in user.chunks]
            embedding = self.model.get_user_embedding(user_chunks)
            user_embeddings.append(embedding.cpu().numpy())
        
        # Get statement embeddings
        statement_embeddings = []
        for statement in test_statements:
            embedding = self.model.get_statement_embedding(statement)
            statement_embeddings.append(embedding.cpu().numpy())
        
        user_embeddings = np.array(user_embeddings)
        statement_embeddings = np.array(statement_embeddings)
        
        # Compute statistics
        all_embeddings = np.vstack([user_embeddings, statement_embeddings])
        
        # Check normalization
        user_norms = np.linalg.norm(user_embeddings, axis=1)
        statement_norms = np.linalg.norm(statement_embeddings, axis=1)
        
        # Compute embedding space statistics
        embedding_mean = np.mean(all_embeddings, axis=0)
        embedding_std = np.std(all_embeddings, axis=0)
        
        return {
            "user_embedding_mean_norm": np.mean(user_norms),
            "user_embedding_std_norm": np.std(user_norms),
            "statement_embedding_mean_norm": np.mean(statement_norms),
            "statement_embedding_std_norm": np.std(statement_norms),
            "embedding_space_mean": np.mean(embedding_mean),
            "embedding_space_std": np.mean(embedding_std),
            "num_user_embeddings": len(user_embeddings),
            "num_statement_embeddings": len(statement_embeddings)
        }
    
    def generate_evaluation_report(self, 
                                 test_users: List[UserData],
                                 candidate_statements: List[str],
                                 test_statements: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            test_users: List of test users
            candidate_statements: List of candidate statements for retrieval
            test_statements: List of statements for zero-shot inference
            
        Returns:
            Comprehensive evaluation report
        """
        print("Generating comprehensive evaluation report...")
        
        report = {}
        
        # Preference retrieval evaluation
        report["preference_retrieval"] = self.evaluate_preference_retrieval(
            test_users, candidate_statements
        )
        
        # Zero-shot inference evaluation
        report["zero_shot_inference"] = self.evaluate_zero_shot_inference(
            test_users, test_statements
        )
        
        # User similarity evaluation
        report["user_similarity"] = self.evaluate_user_similarity(test_users)
        
        # Embedding space analysis
        report["embedding_space"] = self.analyze_embedding_space(test_users, test_statements)
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION REPORT SUMMARY")
        print("="*50)
        
        print(f"\nPreference Retrieval (Top-10):")
        print(f"  Precision: {report['preference_retrieval']['precision@k']:.4f}")
        print(f"  Recall: {report['preference_retrieval']['recall@k']:.4f}")
        print(f"  F1: {report['preference_retrieval']['f1@k']:.4f}")
        
        print(f"\nZero-Shot Inference:")
        print(f"  Precision: {report['zero_shot_inference']['precision']:.4f}")
        print(f"  Recall: {report['zero_shot_inference']['recall']:.4f}")
        print(f"  F1: {report['zero_shot_inference']['f1']:.4f}")
        print(f"  AUC: {report['zero_shot_inference']['auc']:.4f}")
        
        print(f"\nUser Similarity:")
        print(f"  Mean Similarity: {report['user_similarity']['mean_similarity']:.4f}")
        print(f"  Std Similarity: {report['user_similarity']['std_similarity']:.4f}")
        
        print(f"\nEmbedding Space:")
        print(f"  User Embedding Norm: {report['embedding_space']['user_embedding_mean_norm']:.4f}")
        print(f"  Statement Embedding Norm: {report['embedding_space']['statement_embedding_mean_norm']:.4f}")
        
        return report 