import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional

from marlite.algorithm.model.model_config import ModelConfig
from marlite.algorithm.critic.mixer import Mixer
from marlite.algorithm.model import MaskedModel, HyperNetwork


class GroupConsensusMixer(Mixer):
    def __init__(
        self,
        feature_extractor_config: ModelConfig,
        consensus_processor_config: ModelConfig,
        model_config: ModelConfig,
        num_agents: int,
        group_latent_dim: int,
        deterministic_eval: bool = True,
        consensus_mode: str = "vae",
    ):
        super().__init__()
        self.feature_extractor = feature_extractor_config.get_model()
        self.consensus_processor = consensus_processor_config.get_model()
        self.model = model_config.get_model()
        self.num_agents = num_agents
        self.group_latent_dim = group_latent_dim
        self.deterministic_eval = deterministic_eval
        self.consensus_mode = consensus_mode

        if isinstance(self.feature_extractor, MaskedModel):
            self.fe_class_name = "MaskedModel"
        else:
            self.fe_class_name = "Other"

    def forward(
        self,
        q_value_from_agents: torch.Tensor,
        states: torch.Tensor,
        alive_mask: torch.Tensor,
        padding_mask: torch.Tensor,
        group_mu: Optional[torch.Tensor] = None,
        group_log_var: Optional[torch.Tensor] = None,
        group_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        bs = q_value_from_agents.shape[0]
        device = q_value_from_agents.device
        alive_mask_last = alive_mask[:, -1, :]
        states_last = states[:, -1, :].reshape(bs, -1)

        if self.fe_class_name == "MaskedModel":
            state_features = self.feature_extractor(states_last, alive_mask_last)
        else:
            state_features = self.feature_extractor(states_last)

        # Build group consensus input: deduplicate per-agent group distributions
        if group_mu is not None and group_log_var is not None and group_indices is not None:
            all_group_consensus = torch.zeros(
                bs, self.num_agents * self.group_latent_dim, device=device
            )
            for b in range(bs):
                unique_groups = torch.unique(group_indices[b].cpu())
                unique_groups = unique_groups[unique_groups >= 0]
                offset = 0
                for z in unique_groups:
                    mu_z = group_mu[b, z]

                    if self.consensus_mode == "ae":
                        sample_z = mu_z
                    else:
                        log_var_z = group_log_var[b, z]
                        std_z = torch.exp(0.5 * log_var_z)

                        deterministic = self.deterministic_eval and not self.training
                        if deterministic:
                            sample_z = mu_z
                        else:
                            eps = torch.randn_like(std_z)
                            sample_z = mu_z + eps * std_z

                    if offset < self.num_agents * self.group_latent_dim:
                        end = offset + self.group_latent_dim
                        all_group_consensus[b, offset:end] = sample_z
                        offset = end
        else:
            all_group_consensus = torch.zeros(
                bs, self.num_agents * self.group_latent_dim, device=device
            )

        # Consensus processor is a HyperNetwork: (all_group_consensus, state_features) -> group_hidden
        if isinstance(self.consensus_processor, HyperNetwork):
            group_hidden = self.consensus_processor(all_group_consensus, state_features)
        else:
            group_hidden = self.consensus_processor(all_group_consensus)

        # QMix model: (q_values, group_hidden) -> q_tot
        masked_q_values = q_value_from_agents * alive_mask_last
        q_tot = self.model(masked_q_values, group_hidden)

        return {"q_tot": q_tot, "state_features": group_hidden}
