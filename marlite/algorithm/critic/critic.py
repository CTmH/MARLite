import torch.nn as nn
import torch
from marlite.algorithm.model.masked_model import MaskedModel

class Critic(nn.Module):
    """
    Critic network that combines agent Q-values and state features to compute total Q-value.

    Args:
        critic_model: A module that takes q_values and state_features as input
        feature_extractor: A module that extracts state representation from raw states
    """

    def __init__(self, critic_model: nn.Module, feature_extractor: nn.Module):
        super(Critic, self).__init__()
        self.critic_model = critic_model
        self.feature_extractor = feature_extractor
        if isinstance(self.feature_extractor, MaskedModel):
            self.fe_class_name = 'MaskedModel'
        else:
            self.fe_class_name = 'Other'

    def forward(
        self,
        q_value_from_agents: torch.Tensor,
        states: torch.Tensor,
        alive_mask: torch.Tensor
    ) -> dict:
        """
        Forward pass of the critic network.

        Args:
            q_value_from_agents: Tensor of shape [batch_size, n_agents] containing per-agent Q-values
            states: Tensor of shape [batch_size, n_agents, state_dim] containing agent states
            alive_mask: Boolean mask of shape [batch_size, n_agents] indicating which agents are alive

        Returns:
            Dict with keys:
                - "q_tot": Total Q-value for the whole system
                - "state_features": Extracted state features
        """
        # Apply mask to Q-values (zero out dead agents)
        masked_q_values = q_value_from_agents * alive_mask

        # Extract state features using appropriate method
        if self.fe_class_name == 'MaskedModel':
            state_features = self.feature_extractor(states, alive_mask)
        else:
            state_features = self.feature_extractor(states)

        # Compute total Q-value
        q_tot = self.critic_model(masked_q_values, state_features)

        return {
            "q_tot": q_tot,
            "state_features": state_features
        }