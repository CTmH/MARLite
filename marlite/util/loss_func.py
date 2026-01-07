import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from typing import Dict, Type

class PITLoss(_Loss):
    def __init__(self, num_tasks: int, alpha: float = 0.9, eps: float = 1e-8, reduction: str = 'mean'):
        """
        Probability Integral Transformation Loss (PITLoss)

        This loss function normalizes task losses using exponential moving averages
        and transforms them to follow a standard normal distribution.

        Args:
            num_tasks (int): Number of tasks to balance
            alpha (float): Exponential decay rate for moving averages (0.0-1.0)
            eps (float): Small value to prevent division by zero
            reduction (str): Type of loss reduction ('none', 'mean', 'sum')
        """
        super().__init__()
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction

        # Initialize buffers for moving averages
        # Mean is initialized to 0, variance to 0 (not 1 as in original)
        self.register_buffer('moving_mean', torch.zeros(num_tasks))
        self.register_buffer('moving_var', torch.zeros(num_tasks))  # Fixed: was torch.ones()
        self.register_buffer('step', torch.zeros(1, dtype=torch.long))
        self.register_buffer('total_var', torch.zeros(num_tasks))  # For unbiased variance estimation

    def forward(self, losses: torch.Tensor) -> torch.Tensor:
        """
        Compute the PIT loss for multiple tasks.

        Args:
            losses (torch.Tensor): Tensor of task losses with shape (num_tasks,)

        Returns:
            torch.Tensor: PIT loss value
        """
        with torch.no_grad():
            if self.training:
                self.step += 1
                current_losses = losses.detach()

                # Initialize on first step
                if self.step == 1:
                    self.moving_mean.copy_(current_losses)
                else:
                    # Update exponential moving average for mean
                    self.moving_mean.mul_(self.alpha).add_(current_losses, alpha=1-self.alpha)

                    # Calculate difference from current mean
                    diff = current_losses - self.moving_mean

                    # Update exponential moving average for variance (unbiased estimate)
                    self.moving_var.mul_(self.alpha).add_(diff.pow(2), alpha=(1-self.alpha))

        # Calculate unbiased variance estimate
        # This compensates for bias in initial estimates
        unbiased_var = self.moving_var / (1 - self.alpha**self.step)
        std = torch.sqrt(unbiased_var + self.eps)

        # Correct normalization: (x - μ) / σ
        normalized_losses = (losses - self.moving_mean) / std

        # Calculate standard normal CDF values
        sqrt2 = torch.sqrt(torch.tensor(2.0, device=losses.device))
        cdf_values = 0.5 * (1 + torch.erf(normalized_losses / sqrt2))

        # PIT loss: (CDF - 0.5)^2
        pit_loss = (cdf_values - 0.5).pow(2)

        # Apply reduction
        if self.reduction == 'mean':
            return pit_loss.mean()
        elif self.reduction == 'sum':
            return pit_loss.sum()
        else:  # 'none'
            return pit_loss


class InfoNCELoss(_Loss):
    """
    InfoNCE (Info Noise Contrastive Estimation) Loss

    This loss is commonly used in contrastive learning and self-supervised learning.
    It maximizes the similarity between positive pairs while minimizing similarity
    between negative pairs.
    """
    def __init__(self, temperature: float = 0.1, reduction: str = 'mean'):
        """
        Initialize InfoNCE Loss

        Parameters:
            temperature (float): Temperature parameter for scaling logits
            reduction (str): Type of loss reduction ('none', 'mean', 'sum')
        """
        super().__init__(reduction=reduction)
        self.temperature = temperature

    def forward(self, query: torch.Tensor, positive: torch.Tensor, negatives: torch.Tensor = None) -> torch.Tensor:
        """
        Compute InfoNCE loss

        Parameters:
            query (torch.Tensor): Query embeddings (batch_size, embedding_dim)
            positive (torch.Tensor): Positive embeddings (batch_size, embedding_dim)
            negatives (torch.Tensor, optional): Negative embeddings (num_negatives, embedding_dim)
                                              If None, uses all other samples in batch as negatives

        Returns:
            torch.Tensor: InfoNCE loss value
        """
        batch_size = query.size(0)

        # Compute similarity scores
        if negatives is None:
            # Use all other samples in batch as negatives
            # Compute similarity matrix between all queries and all positives
            similarity_matrix = F.cosine_similarity(query.unsqueeze(1), positive.unsqueeze(0), dim=2)

            # Positive pairs are on the diagonal
            positive_similarity = torch.diag(similarity_matrix)

            # Create mask to exclude positive pairs
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=query.device)
            negative_similarities = similarity_matrix[mask].view(batch_size, batch_size - 1)

        else:
            # Use provided negatives
            num_negatives = negatives.size(0)

            # Compute positive similarity
            positive_similarity = F.cosine_similarity(query, positive, dim=1)

            # Compute negative similarities
            negative_similarities = F.cosine_similarity(
                query.unsqueeze(1).expand(-1, num_negatives, -1).reshape(-1, query.size(-1)),
                negatives.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, negatives.size(-1)),
                dim=1
            ).view(batch_size, num_negatives)

        # Scale by temperature
        positive_similarity = positive_similarity / self.temperature
        negative_similarities = negative_similarities / self.temperature

        # Concatenate positive and negative similarities
        logits = torch.cat([positive_similarity.unsqueeze(1), negative_similarities], dim=1)

        # Labels: 0 for positive (first position)
        labels = torch.zeros(batch_size, dtype=torch.long, device=query.device)

        # Compute cross entropy loss
        loss = F.cross_entropy(logits, labels, reduction='none')

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ChamferDistanceLoss(_Loss):
    """
    Computes Chamfer Distance loss between two sets of points.
    Input tensors should have shape: (batch_size, n_points, feature_dim)

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output.
            Default: 'mean'. Options: 'none', 'sum'
        use_squared_distance (bool, optional): Whether to use squared Euclidean distance.
            Default: True (recommended for better gradient behavior)
    """
    def __init__(self, reduction='mean', use_squared_distance=True):
        super().__init__(reduction=reduction)
        self.use_squared_distance = use_squared_distance

    def forward(self, pred_set:torch.Tensor, target_set:torch.Tensor, mask:torch.Tensor=None):
        """
        Args:
            pred_set: Predicted set of points (B, N, D)
            target_set: Target set of points (B, N, D)
            mask: Optional boolean mask indicating valid points (B, N)
        Returns:
            loss: Chamfer Distance loss (scalar tensor when reduction='mean' or 'sum',
                  or (B,) tensor when reduction='none')
        """
        B, N, D = pred_set.shape

        # Compute pairwise distances
        dist_matrix = self._pairwise_distance(pred_set, target_set)  # Removed mask parameter

        # Compute min distances in both directions
        min_dist_pred_to_target = dist_matrix.min(dim=2).values  # (B, N)
        min_dist_target_to_pred = dist_matrix.min(dim=1).values  # (B, N)

        # Apply mask if provided
        if mask is not None:
            valid_pred = mask.float()
            valid_target = mask.float()
            min_dist_pred_to_target = min_dist_pred_to_target * valid_pred
            min_dist_target_to_pred = min_dist_target_to_pred * valid_target

            # Normalize by number of valid points per set
            num_valid_pred = valid_pred.sum(dim=1, keepdim=True).clamp_min(1)
            num_valid_target = valid_target.sum(dim=1, keepdim=True).clamp_min(1)

            forward_loss = min_dist_pred_to_target.sum(dim=1) / num_valid_pred.squeeze()
            backward_loss = min_dist_target_to_pred.sum(dim=1) / num_valid_target.squeeze()

            loss_per_batch = forward_loss + backward_loss
        else:
            # Normalize by number of points
            num_points = torch.tensor(N, dtype=torch.float32, device=pred_set.device)
            forward_loss = min_dist_pred_to_target.sum(dim=1) / num_points
            backward_loss = min_dist_target_to_pred.sum(dim=1) / num_points

            loss_per_batch = forward_loss + backward_loss

        # Apply reduction
        if self.reduction == 'none':
            return loss_per_batch # (B,)
        elif self.reduction == 'mean':
            return loss_per_batch.mean()
        elif self.reduction == 'sum':
            return loss_per_batch.sum()
        else:
            raise ValueError(f"Invalid reduction type: {self.reduction}")

    def _pairwise_distance(self, set1:torch.Tensor, set2:torch.Tensor):
        """
        Compute pairwise Euclidean distances between two sets of points.

        Args:
            set1: (B, N1, D)
            set2: (B, N2, D)
        Returns:
            dist_matrix: (B, N1, N2)
        """
        # Compute squared differences
        diff = set1.unsqueeze(2) - set2.unsqueeze(1)  # (B, N1, 1, D) - (B, 1, N2, D) -> (B, N1, N2, D)

        # Compute squared Euclidean distance
        dist_sq = (diff ** 2).sum(dim=-1)  # (B, N1, N2)

        if self.use_squared_distance:
            return dist_sq
        else:
            return dist_sq.clamp_min(1e-12).sqrt()


REGISTERED_RECONSTRUCTION_LOSS: Dict[str, Type[_Loss]] = {
    "ChamferDist": ChamferDistanceLoss
}