import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

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