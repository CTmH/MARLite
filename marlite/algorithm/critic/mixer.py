import torch.nn as nn
import torch
from typing import Dict
from marlite.algorithm.model import ModelConfig
from marlite.algorithm.model import RNNModel, Conv1DModel, AttentionModel, MaskedModel


class Mixer(nn.Module):
    """
    Base class for mixer networks that combine per-agent Q-values with global state information
    to compute a centralized total Q-value. Subclasses must implement the forward method.
    """

    def __init__(self):
        super(Mixer, self).__init__()

    def forward(
        self,
        q_value_from_agents: torch.Tensor,
        states: torch.Tensor,
        alive_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the mixer network.

        Args:
            q_value_from_agents: Tensor of shape [batch_size, n_agents] containing per-agent Q-values
            states: Tensor of shape [batch_size, seq_len, n_agents, state_dim] containing agent states over time
            alive_mask: Boolean mask of shape [batch_size, seq_len, n_agents] indicating which agents are alive at each timestep
            padding_mask: Boolean mask of shape [batch_size, seq_len] indicating valid timesteps (non-padded)

        Returns:
            Dictionary containing:
                - "q_tot": Total Q-value for the system
                - "state_features": Extracted/global state features
                - Additional outputs like mu/std in probabilistic variants
        """
        raise NotImplementedError


class QMixer(Mixer):
    """
    Standard QMixer network that combines agent Q-values and extracted state features
    to compute a total Q-value using a base mixing model.

    The feature extractor processes the final timestep's state, and the base model mixes
    masked Q-values with these global features to produce a centralized value estimate.
    """

    def __init__(self, base_model_config: ModelConfig, feature_extractor_config: ModelConfig):
        super(QMixer, self).__init__()
        self.base_model = base_model_config.get_model()
        self.feature_extractor = feature_extractor_config.get_model()
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
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the QMixer network.

        Args:
            q_value_from_agents: Tensor of shape [batch_size, n_agents] containing per-agent Q-values
            states: Tensor of shape [batch_size, seq_len, n_agents, state_dim] containing agent states
            alive_mask: Boolean mask of shape [batch_size, seq_len, n_agents] for alive agents
            padding_mask: Boolean mask of shape [batch_size, seq_len] for valid timesteps

        Returns:
            Dict with keys:
                - "q_tot": Total Q-value for the whole system
                - "state_features": Extracted state features from the final timestep
        """
        # Use only the last timestep's data
        alive_mask = alive_mask[:, -1, :]  # (B, T, N) -> (B, N)
        states = states[:, -1, :]  # (B, T, N, F) -> (B, N, F)

        # Extract state features, applying agent masking if supported
        if self.fe_class_name == 'MaskedModel':
            encoded_states = self.feature_extractor(states, alive_mask)
        else:
            encoded_states = self.feature_extractor(states)

        # Mask out Q-values of dead agents
        masked_q_values = q_value_from_agents * alive_mask

        # Compute total Q-value using base mixing model
        q_tot = self.base_model(masked_q_values, encoded_states)

        return {
            "q_tot": q_tot,
            "state_features": encoded_states
        }


class SeqQMixer(Mixer):
    """
    Sequential QMixer that processes temporal sequences of states using a sequence model
    (RNN, CNN, or Attention) before mixing with Q-values.

    This allows the critic to capture temporal dynamics in the environment state before
    computing the total Q-value.
    """

    def __init__(
        self,
        base_model_config: ModelConfig,
        feature_extractor_config: ModelConfig,
        seq_model_config: ModelConfig,
        state_feature_type: str = "Seq"
    ):
        """
        Initialize the SeqQMixer.

        Args:
            base_model_config: Configuration for the final mixing network
            feature_extractor_config: Configuration for per-timestep state feature extraction
            seq_model_config: Configuration for the sequential processing model
            state_feature_type: Type of state features to return ('Seq' for full sequence output,
                               'State' for last timestep features only)
        """
        super(SeqQMixer, self).__init__()
        self.base_model = base_model_config.get_model()
        self.feature_extractor = feature_extractor_config.get_model()
        self.seq_model = seq_model_config.get_model()
        self.state_feature_type = state_feature_type

        if isinstance(self.feature_extractor, MaskedModel):
            self.fe_class_name = 'MaskedModel'
        else:
            self.fe_class_name = 'Other'

        if isinstance(self.seq_model, RNNModel):
            self.seq_model_class_name = 'RNNModel'
        elif isinstance(self.seq_model, Conv1DModel):
            self.seq_model_class_name = 'Conv1DModel'
        elif isinstance(self.seq_model, AttentionModel):
            self.seq_model_class_name = 'AttentionModel'
        else:
            self.seq_model_class_name = self.seq_model.__class__.__name__

    def forward(
        self,
        q_value_from_agents: torch.Tensor,
        states: torch.Tensor,
        alive_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the sequential QMixer.

        Args:
            q_value_from_agents: Tensor of shape [batch_size, n_agents] containing per-agent Q-values
            states: Tensor of shape [batch_size, seq_len, n_agents, state_dim] containing agent states
            alive_mask: Boolean mask of shape [batch_size, seq_len, n_agents] for alive agents
            padding_mask: Boolean mask of shape [batch_size, seq_len] for valid timesteps

        Returns:
            Dict with keys:
                - "q_tot": Total Q-value for the whole system
                - "state_features": Sequence-aware state features based on state_feature_type
        """
        bs = q_value_from_agents.shape[0]
        ts = states.shape[1]
        state_shape = states.shape[2:]
        states = states.reshape(bs * ts, *state_shape)

        # Extract features for each timestep
        if self.fe_class_name == 'MaskedModel':
            encoded_states = self.feature_extractor(states, alive_mask.reshape(bs * ts, -1))
        else:
            encoded_states = self.feature_extractor(states)

        encoded_states = encoded_states.reshape(bs, ts, -1)
        last_encoded_states = encoded_states[:, -1, :]

        # Process sequence with appropriate model
        if self.seq_model_class_name == 'Conv1DModel':
            # Transpose for Conv1D: (B, T, F) -> (B, F, T)
            encoded_states = encoded_states.permute(0, 2, 1)
            hidden_states = self.seq_model(encoded_states)
        elif self.seq_model_class_name == 'RNNModel':
            hidden_states = self.seq_model(encoded_states)
        elif self.seq_model_class_name == 'AttentionModel':
            hidden_states = self.seq_model(encoded_states, padding_mask)
        else:
            # Default: use only last timestep
            hidden_states = self.seq_model(encoded_states[:, -1, :])

        # Apply agent masking to Q-values
        masked_q_values = q_value_from_agents * alive_mask[:, -1, :]

        # Compute total Q-value
        q_tot = self.base_model(masked_q_values, hidden_states)

        # Select appropriate state features to return
        if self.state_feature_type == 'State':
            state_features = last_encoded_states
        elif self.state_feature_type == 'Seq':
            state_features = hidden_states
        else:
            state_features = None

        return {
            "q_tot": q_tot,
            "state_features": state_features
        }


class ProbQMixer(Mixer):
    """
    Probabilistic QMixer that models uncertainty in state representation through
    variational inference. The feature extractor outputs mean and log-variance parameters
    of a Gaussian distribution, enabling stochastic sampling of state features.
    """

    def __init__(self, base_model_config: ModelConfig, feature_extractor_config: ModelConfig, deterministic_eval = True):
        super(ProbQMixer, self).__init__()
        self.base_model = base_model_config.get_model()
        self.feature_extractor = feature_extractor_config.get_model()
        if isinstance(self.feature_extractor, MaskedModel):
            self.fe_class_name = 'MaskedModel'
        else:
            self.fe_class_name = 'Other'

        self.deterministic_eval = deterministic_eval

    def forward(
        self,
        q_value_from_agents: torch.Tensor,
        states: torch.Tensor,
        alive_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the probabilistic QMixer.

        Args:
            q_value_from_agents: Tensor of shape [batch_size, n_agents] containing per-agent Q-values
            states: Tensor of shape [batch_size, seq_len, n_agents, state_dim] containing agent states
            alive_mask: Boolean mask of shape [batch_size, seq_len, n_agents] for alive agents
            padding_mask: Boolean mask of shape [batch_size, seq_len] for valid timesteps

        Returns:
            Dict with keys:
                - "q_tot": Total Q-value computed using sampled state features
                - "state_features": Concatenated mean and log-variance tensors
                - "mu": Mean of the latent state distribution
                - "std": Standard deviation of the latent state distribution
        """
        alive_mask = alive_mask[:, -1, :]  # Use last timestep
        states = states[:, -1, :]  # (B, T, N, F) -> (B, N, F)

        # Extract probabilistic state features
        if self.fe_class_name == 'MaskedModel':
            encoded_states = self.feature_extractor(states, alive_mask)
        else:
            encoded_states = self.feature_extractor(states)

        # Split into mean and log-variance components
        dim = encoded_states.size(-1) // 2
        mu = encoded_states[:, :dim]  # Mean
        log_var = encoded_states[:, dim:]  # Log variance
        std = torch.exp(0.5 * log_var)

        # Use deterministic evaluation if enabled and not in training mode
        if self.deterministic_eval and not self.training:
            # Directly use mu as output without reparameterization sampling
            sample = mu
        else:
            # Reparameterization trick for gradient estimation
            eps = torch.randn_like(std)
            sample = mu + eps * std  # Sample from N(mu, sigma^2)

        # Mask Q-values and compute total Q-value
        masked_q_values = q_value_from_agents * alive_mask
        q_tot = self.base_model(masked_q_values, sample)

        return {
            "q_tot": q_tot,
            "state_features": encoded_states,
            "mu": mu,
            "std": std
        }


class ProbSeqQMixer(Mixer):
    """
    Probabilistic Sequential QMixer that combines temporal modeling with uncertainty
    estimation. Processes state sequences through a sequential model, then applies
    variational inference to the final hidden state.
    """

    def __init__(
        self,
        base_model_config: ModelConfig,
        feature_extractor_config: ModelConfig,
        seq_model_config: ModelConfig,
        deterministic_eval = True
    ):
        """
        Initialize the Probabilistic Sequential QMixer.

        Args:
            base_model_config: Configuration for the final mixing network
            feature_extractor_config: Configuration for per-timestep feature extraction
            seq_model_config: Configuration for the sequential processing model
        """
        super(ProbSeqQMixer, self).__init__()
        self.base_model = base_model_config.get_model()
        self.feature_extractor = feature_extractor_config.get_model()
        self.seq_model = seq_model_config.get_model()

        self.deterministic_eval = deterministic_eval

        if isinstance(self.feature_extractor, MaskedModel):
            self.fe_class_name = 'MaskedModel'
        else:
            self.fe_class_name = 'Other'

        if isinstance(self.seq_model, RNNModel):
            self.seq_model_class_name = 'RNNModel'
        elif isinstance(self.seq_model, Conv1DModel):
            self.seq_model_class_name = 'Conv1DModel'
        elif isinstance(self.seq_model, AttentionModel):
            self.seq_model_class_name = 'AttentionModel'
        else:
            self.seq_model_class_name = self.seq_model.__class__.__name__

    def forward(
        self,
        q_value_from_agents: torch.Tensor,
        states: torch.Tensor,
        alive_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the probabilistic sequential QMixer.

        Args:
            q_value_from_agents: Tensor of shape [batch_size, n_agents] containing per-agent Q-values
            states: Tensor of shape [batch_size, seq_len, n_agents, state_dim] containing agent states
            alive_mask: Boolean mask of shape [batch_size, seq_len, n_agents] for alive agents
            padding_mask: Boolean mask of shape [batch_size, seq_len] for valid timesteps

        Returns:
            Dict with keys:
                - "q_tot": Total Q-value computed using sampled sequence features
                - "state_features": Concatenated mean and log-variance of final hidden state
                - "mu": Mean of the latent sequence representation
                - "std": Standard deviation of the latent sequence representation
        """
        bs = q_value_from_agents.shape[0]
        ts = states.shape[1]
        state_shape = states.shape[2:]
        states = states.reshape(bs * ts, *state_shape)

        # Extract features for each timestep
        if self.fe_class_name == 'MaskedModel':
            encoded_states = self.feature_extractor(states, alive_mask.reshape(bs * ts, -1))
        else:
            encoded_states = self.feature_extractor(states)

        encoded_states = encoded_states.reshape(bs, ts, -1)

        # Process sequence with appropriate model
        if self.seq_model_class_name == 'Conv1DModel':
            encoded_states = encoded_states.permute(0, 2, 1)  # (B, T, F) -> (B, F, T)
            hidden_states = self.seq_model(encoded_states)
        elif self.seq_model_class_name == 'RNNModel':
            hidden_states = self.seq_model(encoded_states)
        elif self.seq_model_class_name == 'AttentionModel':
            hidden_states = self.seq_model(encoded_states, padding_mask)
        else:
            hidden_states = self.seq_model(encoded_states[:, -1, :])

        # Split final hidden state into mean and log-variance
        dim = hidden_states.size(-1) // 2
        mu = hidden_states[:, :dim]  # Mean
        log_var = hidden_states[:, dim:]  # Log variance
        std = torch.exp(0.5 * log_var)

        # Use deterministic evaluation if enabled and not in training mode
        if self.deterministic_eval and not self.training:
            # Directly use mu as output without reparameterization sampling
            sample = mu
        else:
            # Reparameterization trick
            eps = torch.randn_like(std)
            sample = mu + eps * std  # Sample from learned distribution

        # Mask Q-values and compute total Q-value
        masked_q_values = q_value_from_agents * alive_mask[:, -1, :]
        q_tot = self.base_model(masked_q_values, sample)

        return {
            "q_tot": q_tot,
            "state_features": hidden_states,
            "mu": mu,
            "std": std
        }