"""
Contrastive Loss Functions for Universal User Embeddings.

This implements the symmetric InfoNCE loss described in the paper:
- Maximizes similarity between users and their positive statements
- Minimizes similarity between users and negative statements
- Uses temperature scaling for better training dynamics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class ContrastiveLoss(nn.Module):
    """
    Symmetric contrastive loss for Universal User Embeddings.
    
    This implements the InfoNCE loss described in the paper:
    L = (1/B) * sum_i [-log(exp(sim(u_i, s_i^+) / τ) / sum_j exp(sim(u_i, s_j^+) / τ)) 
                       -log(exp(sim(s_i^+, u_i) / τ) / sum_j exp(sim(s_i^+, u_j) / τ))]
    
    where:
    - u_i is the user embedding
    - s_i^+ is the positive statement embedding for user i
    - τ is the temperature parameter
    - B is the batch size
    """
    
    def __init__(self, temperature: float = 0.1, learnable_temperature: bool = True):
        super().__init__()
        self.temperature = temperature
        self.learnable_temperature = learnable_temperature
        
        if learnable_temperature:
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        else:
            self.log_temperature = None
    
    def forward(self, 
                user_embeddings: torch.Tensor,
                positive_embeddings: torch.Tensor,
                negative_embeddings: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute symmetric contrastive loss.
        
        Args:
            user_embeddings: (batch_size, embedding_dim) - L2 normalized user embeddings
            positive_embeddings: (batch_size, embedding_dim) - L2 normalized positive statement embeddings
            negative_embeddings: Optional (batch_size, num_negatives, embedding_dim) - L2 normalized negative embeddings
            
        Returns:
            Dictionary containing loss and metrics
        """
        batch_size = user_embeddings.size(0)
        
        # Get temperature
        if self.learnable_temperature:
            temperature = torch.exp(self.log_temperature)
        else:
            temperature = self.temperature
        
        # Compute similarity matrix between users and positive statements
        # sim_matrix[i][j] = similarity between user i and positive statement j
        sim_matrix = torch.matmul(user_embeddings, positive_embeddings.t()) / temperature
        
        # Create labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=user_embeddings.device)
        
        # Compute loss in both directions (symmetric)
        loss_user_to_statement = F.cross_entropy(sim_matrix, labels)
        loss_statement_to_user = F.cross_entropy(sim_matrix.t(), labels)
        
        # Symmetric loss
        loss = (loss_user_to_statement + loss_statement_to_user) / 2
        
        # Compute accuracy
        with torch.no_grad():
            # User to statement accuracy
            user_to_statement_preds = torch.argmax(sim_matrix, dim=1)
            user_to_statement_acc = (user_to_statement_preds == labels).float().mean()
            
            # Statement to user accuracy
            statement_to_user_preds = torch.argmax(sim_matrix.t(), dim=1)
            statement_to_user_acc = (statement_to_user_preds == labels).float().mean()
            
            # Average accuracy
            avg_acc = (user_to_statement_acc + statement_to_user_acc) / 2
        
        result = {
            "loss": loss,
            "temperature": temperature,
            "user_to_statement_loss": loss_user_to_statement,
            "statement_to_user_loss": loss_statement_to_user,
            "user_to_statement_acc": user_to_statement_acc,
            "statement_to_user_acc": statement_to_user_acc,
            "avg_acc": avg_acc
        }
        
        # Add negative mining loss if negative embeddings are provided
        if negative_embeddings is not None:
            neg_loss = self._compute_negative_loss(
                user_embeddings, positive_embeddings, negative_embeddings, temperature
            )
            result["negative_loss"] = neg_loss
            result["total_loss"] = loss + neg_loss
        
        return result
    
    def _compute_negative_loss(self,
                              user_embeddings: torch.Tensor,
                              positive_embeddings: torch.Tensor,
                              negative_embeddings: torch.Tensor,
                              temperature: float) -> torch.Tensor:
        """
        Compute additional loss using negative samples.
        
        Args:
            user_embeddings: (batch_size, embedding_dim)
            positive_embeddings: (batch_size, embedding_dim)
            negative_embeddings: (batch_size, num_negatives, embedding_dim)
            temperature: Temperature parameter
            
        Returns:
            Negative loss value
        """
        batch_size, num_negatives, _ = negative_embeddings.shape
        
        # Compute similarities between users and their negative statements
        # (batch_size, num_negatives)
        user_neg_sims = torch.bmm(
            user_embeddings.unsqueeze(1),
            negative_embeddings.transpose(1, 2)
        ).squeeze(1) / temperature
        
        # Compute similarities between positive statements and negative statements
        # (batch_size, num_negatives)
        pos_neg_sims = torch.bmm(
            positive_embeddings.unsqueeze(1),
            negative_embeddings.transpose(1, 2)
        ).squeeze(1) / temperature
        
        # Create logits: [positive_sim, negative_sim_1, negative_sim_2, ...]
        logits = torch.cat([
            torch.sum(user_embeddings * positive_embeddings, dim=1, keepdim=True) / temperature,
            user_neg_sims
        ], dim=1)
        
        # Labels are 0 (positive pair should have highest similarity)
        labels = torch.zeros(batch_size, device=user_embeddings.device, dtype=torch.long)
        
        # Compute cross-entropy loss
        neg_loss = F.cross_entropy(logits, labels)
        
        return neg_loss


class HardNegativeMiningLoss(nn.Module):
    """
    Enhanced contrastive loss with hard negative mining.
    
    This loss function identifies the most challenging negative samples
    and focuses training on them, improving the quality of learned embeddings.
    """
    
    def __init__(self, temperature: float = 0.1, mining_ratio: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.mining_ratio = mining_ratio
        self.base_loss = ContrastiveLoss(temperature, learnable_temperature=False)
    
    def forward(self,
                user_embeddings: torch.Tensor,
                positive_embeddings: torch.Tensor,
                negative_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute contrastive loss with hard negative mining.
        
        Args:
            user_embeddings: (batch_size, embedding_dim)
            positive_embeddings: (batch_size, embedding_dim)
            negative_embeddings: (batch_size, num_negatives, embedding_dim)
            
        Returns:
            Dictionary containing loss and metrics
        """
        batch_size, num_negatives, _ = negative_embeddings.shape
        
        # Compute similarities with all negative samples
        user_neg_sims = torch.bmm(
            user_embeddings.unsqueeze(1),
            negative_embeddings.transpose(1, 2)
        ).squeeze(1) / self.temperature
        
        # Find hard negatives (highest similarity scores)
        num_hard_negatives = max(1, int(num_negatives * self.mining_ratio))
        hard_neg_indices = torch.topk(user_neg_sims, k=num_hard_negatives, dim=1)[1]
        
        # Select hard negative embeddings
        batch_indices = torch.arange(batch_size, device=user_embeddings.device).unsqueeze(1).expand(-1, num_hard_negatives)
        hard_negative_embeddings = negative_embeddings[batch_indices, hard_neg_indices]
        
        # Compute base contrastive loss
        base_result = self.base_loss(user_embeddings, positive_embeddings, hard_negative_embeddings)
        
        # Add hard negative mining metrics
        base_result["hard_negative_ratio"] = self.mining_ratio
        base_result["num_hard_negatives"] = num_hard_negatives
        
        return base_result


class TripletLoss(nn.Module):
    """
    Triplet loss variant for Universal User Embeddings.
    
    This loss ensures that the distance between a user and their positive statement
    is smaller than the distance to negative statements by a margin.
    """
    
    def __init__(self, margin: float = 0.3, temperature: float = 0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(self,
                user_embeddings: torch.Tensor,
                positive_embeddings: torch.Tensor,
                negative_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute triplet loss.
        
        Args:
            user_embeddings: (batch_size, embedding_dim)
            positive_embeddings: (batch_size, embedding_dim)
            negative_embeddings: (batch_size, num_negatives, embedding_dim)
            
        Returns:
            Dictionary containing loss and metrics
        """
        batch_size, num_negatives, _ = negative_embeddings.shape
        
        # Compute distances (lower is better for L2 normalized embeddings)
        pos_distances = 2 - 2 * torch.sum(user_embeddings * positive_embeddings, dim=1)
        
        # Compute distances to negative samples
        neg_distances = 2 - 2 * torch.bmm(
            user_embeddings.unsqueeze(1),
            negative_embeddings.transpose(1, 2)
        ).squeeze(1)
        
        # Find hardest negative for each user
        hardest_neg_distances, _ = torch.min(neg_distances, dim=1)
        
        # Compute triplet loss
        triplet_loss = F.relu(pos_distances - hardest_neg_distances + self.margin).mean()
        
        # Compute accuracy (positive distance < negative distance)
        accuracy = (pos_distances < hardest_neg_distances).float().mean()
        
        return {
            "loss": triplet_loss,
            "accuracy": accuracy,
            "avg_positive_distance": pos_distances.mean(),
            "avg_negative_distance": hardest_neg_distances.mean(),
            "margin": self.margin
        } 