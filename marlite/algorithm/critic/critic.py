import torch.nn as nn
import torch
from marlite.algorithm.model import RNNModel, Conv1DModel, AttentionModel, MaskedModel

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
        alive_mask: torch.Tensor,
        padding_mask: torch.Tensor,
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
        alive_mask = alive_mask[:,-1,:] # (B, T, N) -> (B, N)
        states = states[:,-1,:] # (B, T, F) -> (B, F) Take only the last state in the sequence
        # Extract state features using appropriate method
        if self.fe_class_name == 'MaskedModel':
            encoded_states = self.feature_extractor(states, alive_mask)
        else:
            encoded_states = self.feature_extractor(states)

        # Apply mask to Q-values (zero out dead agents)
        masked_q_values = q_value_from_agents * alive_mask
        # Compute total Q-value
        q_tot = self.critic_model(masked_q_values, encoded_states)

        return {
            "q_tot": q_tot,
            "state_features": encoded_states
        }

class SeqCritic(nn.Module):
    """
    Critic network that combines agent Q-values and state features to compute total Q-value.

    Args:
        critic_model: A module that takes q_values and state_features as input
        feature_extractor: A module that extracts state representation from raw states
    """

    def __init__(self, critic_model: nn.Module, feature_extractor: nn.Module, seq_model: nn.Module, state_features_type: str = "Seq"):
        super(SeqCritic, self).__init__()
        self.critic_model = critic_model
        self.feature_extractor = feature_extractor
        self.seq_model = seq_model
        self.state_features_type = state_features_type

        if isinstance(self.feature_extractor, MaskedModel):
            self.fe_class_name = 'MaskedModel'
        else:
            self.fe_class_name = 'Other'

        if isinstance(seq_model, RNNModel):
            self.seq_model_class_name = 'RNNModel'
        elif isinstance(seq_model, Conv1DModel):
            self.seq_model_class_name = 'Conv1DModel'
        elif isinstance(seq_model, AttentionModel):
            self.seq_model_class_name = 'AttentionModel'
        else:
            self.seq_model_class_name = seq_model.__class__.__name__

    def forward(
        self,
        q_value_from_agents: torch.Tensor,
        states: torch.Tensor,
        alive_mask: torch.Tensor,
        padding_mask: torch.Tensor,
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
        bs = q_value_from_agents.shape[0]
        ts = states.shape[1]
        state_shape = states.shape[2:]
        states = states.reshape(bs*ts, *state_shape)

        # Extract state features using appropriate method
        if self.fe_class_name == 'MaskedModel':
            encoded_states = self.feature_extractor(states, alive_mask.reshape(bs*ts, -1))
        else:
            encoded_states = self.feature_extractor(states)

        encoded_states = encoded_states.reshape(bs, ts, -1)
        last_encoded_states = encoded_states[:, -1, :]

        if self.seq_model_class_name == 'Conv1DModel':
            encoded_states = encoded_states.permute(0, 2, 1) # (B, T, F) -> (B, F, T)
            hidden_states = self.seq_model(encoded_states)
        elif self.seq_model_class_name == 'RNNModel':
            hidden_states = self.seq_model(encoded_states)
        elif self.seq_model_class_name == 'AttentionModel':
            hidden_states = self.seq_model(encoded_states, padding_mask)
        else:
            hidden_states = self.seq_model(encoded_states[:,-1,:])

        # Apply mask to Q-values (zero out dead agents)
        masked_q_values = q_value_from_agents * alive_mask[:,-1,:]
        # Compute total Q-value
        q_tot = self.critic_model(masked_q_values, hidden_states)

        if self.state_features_type == 'State':
            state_features = last_encoded_states
        else:
            state_features = hidden_states
        return {
            "q_tot": q_tot,
            "state_features": state_features
        }