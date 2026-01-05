import torch

def process_probabilistic_output(aggr_output: torch.Tensor, deterministic: bool):
    """Process the aggregated output to get mean and std for probabilistic models."""
    # Split output into mean and log variance along the last dimension
    dim = aggr_output.size(-1) // 2
    mu = aggr_output[..., :dim]  # Mean - preserves all leading dimensions
    log_var = aggr_output[..., dim:]  # Log variance - preserves all leading dimensions
    std = torch.exp(0.5 * log_var)

    # Reparameterization or deterministic sampling based on mode
    if deterministic:
        # During evaluation with deterministic_eval=True, use mu directly
        aggregated_msg = mu
    else:
        # During training or when deterministic_eval=False, use reparameterization
        eps = torch.randn_like(std)
        aggregated_msg = mu + eps * std  # Sample from Gaussian distribution

    return aggregated_msg, mu, std