import torch
from torch.nn.modules.loss import _Loss

class PITLoss(_Loss):
    def __init__(self, num_tasks: int, alpha: float = 0.9, eps: float = 1e-8, reduction: str = 'mean'):
        """
        PITLoss: Probability Integral Transformation Loss

        Parameters:
            num_tasks (int): Number of tasks
            alpha (float): Exponential decay rate for variance estimation (0.0-1.0)
            eps (float): Small value to prevent division by zero
            reduction (str): Type of loss reduction ('none', 'mean', 'sum')
        """
        super().__init__(reduction=reduction)
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.eps = eps

        # Register buffers for moving averages
        self.register_buffer('moving_mean', torch.zeros(num_tasks))
        self.register_buffer('moving_var', torch.ones(num_tasks))
        self.register_buffer('step', torch.zeros(1, dtype=torch.long))

    def forward(self, losses: torch.Tensor) -> torch.Tensor:
        """
        Compute the PIT loss

        Parameters:
            losses (torch.Tensor): Tensor of task losses with shape (num_tasks,)

        Returns:
            torch.Tensor: PIT loss value
        """
        # PyTorch automatically handles device consistency for registered buffers
        # 1. Update variance estimate using exponential moving average (EMA)
        with torch.no_grad():
            if self.training:
                self.step += 1
                current_mean = losses.detach()

                # Initialize on the first step
                if self.step == 1:
                    self.moving_mean.copy_(current_mean)
                else:
                    # Update moving mean
                    self.moving_mean.mul_(self.alpha).add_(current_mean, alpha=1 - self.alpha)
                    diff = current_mean - self.moving_mean
                    # Update moving variance
                    self.moving_var.mul_(self.alpha).addcmul_(diff, diff, value=1 - self.alpha)

        # 2. Calculate normalized losses
        std = torch.sqrt(self.moving_var + self.eps)
        normalized_losses = losses / std

        # 3. Calculate CDF values (standard normal distribution)
        # Create constant tensor on same device as losses
        sqrt2 = torch.sqrt(torch.tensor(2.0, device=losses.device))
        cdf_values = 0.5 * (1 + torch.erf(normalized_losses / sqrt2))

        # 4. Calculate the base PIT loss
        pit_loss = (cdf_values - 0.5) ** 2

        # 5. Apply reduction
        if self.reduction == 'mean':
            return pit_loss.mean()
        elif self.reduction == 'sum':
            return pit_loss.sum()
        else:  # 'none'
            return pit_loss