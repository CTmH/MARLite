import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from absl import logging
from typing import Dict, Any, List, Optional
from marlite.algorithm.model.model_config import ModelConfig
from marlite.algorithm.model import RNNModel, Conv1DModel, AttentionModel
from marlite.algorithm.agents.agent_group import AgentGroup
from marlite.algorithm.group_builder import GroupBuilderConfig
from marlite.util.prob_util import process_probabilistic_output


class GroupConsensusAgentGroup(AgentGroup):
    def __init__(
        self,
        agent_model_dict: Dict[str, str],
        feature_extractor_configs: Dict[str, ModelConfig],
        group_estimate_feature_extractor_configs: Dict[str, ModelConfig],
        encoder_configs: Dict[str, ModelConfig],
        decoder_configs: Dict[str, ModelConfig],
        group_builder_config: GroupBuilderConfig,
        deterministic_eval: bool = True,
        enable_rl_grad_to_group_estimate: bool = False,
        merge_mode: str = "bayesian",
        consensus_mode: str = "vae",
    ) -> None:
        super().__init__()
        self.agent_model_dict = agent_model_dict
        self.deterministic_eval = deterministic_eval
        self.enable_rl_grad_to_group_estimate = enable_rl_grad_to_group_estimate
        if merge_mode not in ("sample_mean", "bayesian"):
            raise ValueError(
                f"merge_mode must be 'sample_mean' or 'bayesian', got '{merge_mode}'"
            )
        self.merge_mode = merge_mode
        if consensus_mode not in ("vae", "ae"):
            raise ValueError(
                f"consensus_mode must be 'vae' or 'ae', got '{consensus_mode}'"
            )
        self.consensus_mode = consensus_mode

        self.feature_extractors = nn.ModuleDict()
        for model_name, config in feature_extractor_configs.items():
            self.feature_extractors[model_name] = config.get_model()

        self.group_estimate_feature_extractors = nn.ModuleDict()
        for model_name, config in group_estimate_feature_extractor_configs.items():
            self.group_estimate_feature_extractors[model_name] = config.get_model()

        self.encoders = nn.ModuleDict()
        for model_name, config in encoder_configs.items():
            self.encoders[model_name] = config.get_model()

        self.decoders = nn.ModuleDict()
        for model_name, config in decoder_configs.items():
            self.decoders[model_name] = config.get_model()

        self.add_module("group_builder", group_builder_config.get_group_builder())

        self.model_to_agents = {model_name: [] for model_name in encoder_configs.keys()}
        self.model_to_agent_indices = {
            model_name: [] for model_name in encoder_configs.keys()
        }
        for i, (agent_name, model_name) in enumerate(self.agent_model_dict.items()):
            assert model_name in self.model_to_agents.keys(), (
                f"Model {model_name} not found in model_configs"
            )
            self.model_to_agents[model_name].append(agent_name)
            self.model_to_agent_indices[model_name].append(i)

        # Identify model class for temporal dispatch (same pattern as GraphAgentGroup)
        self.model_class_names = {}
        for model_name, model in self.encoders.items():
            if isinstance(model, RNNModel):
                self.model_class_names[model_name] = "RNNModel"
            elif isinstance(model, Conv1DModel):
                self.model_class_names[model_name] = "Conv1DModel"
            elif isinstance(model, AttentionModel):
                self.model_class_names[model_name] = "AttentionModel"
            else:
                self.model_class_names[model_name] = model.__class__.__name__

    def _process_observations(self, observations, traj_padding_mask):
        """Process observations through dual-path pipeline.

        Path 1 — group_estimate_feature_extractor (ge_fe):
            Always operates on last timestep only.  The latent distribution
            (μ, log σ²) represents the agent's current belief state, so
            only the most recent observation is needed.

        Path 2 — feature_extractor (fe) + encoder (enc):
            Dispatches by model_class_names to handle temporal models
            (RNN, Conv1D, Attention) or falls back to last-timestep MLP.

        Args:
            observations:  (B, N, T, *(obs_shape))   float tensor
            traj_padding_mask:  (B, N, T)   bool  (agents × timesteps)

        Returns:
            agent_latent:  (B, N, 2*L)    [μ | log σ²]  concatenated
            local_obs:     (B, N, F)      temporally-processed local features
        """
        agent_latent = [None for _ in range(len(self.agent_model_dict))]
        local_obs    = [None for _ in range(len(self.agent_model_dict))]

        for model_name in self.feature_extractors.keys():
            fe    = self.feature_extractors[model_name]
            ge_fe = self.group_estimate_feature_extractors[model_name]
            enc   = self.encoders[model_name]
            idx   = self.model_to_agent_indices[model_name]
            model_class_name = self.model_class_names[model_name]

            # observation shape: (B, N, T, *(obs_shape))
            obs = observations[:, idx]          # (B, n_agents, T, *(obs_shape))
            bs = obs.shape[0]
            n_agents = len(idx)
            ts = obs.shape[2]
            obs_shape = list(obs.shape[3:])

            # ── Path 1: ge_fe — always last timestep ─────────────────────
            last_obs = obs[:, :, -1, :]                          # (B, n_agents, *(obs_shape))
            last_obs = last_obs.reshape(bs * n_agents, *obs_shape).to(self.device)
            #   (B, n_agents, *(obs_shape)) → (B*n_agents, *(obs_shape))

            agent_latent_selected = ge_fe(last_obs)
            #   (B*n_agents, *(obs_shape)) → (B*n_agents, 2*L)
            agent_latent_selected = agent_latent_selected.reshape(bs, n_agents, -1)
            #   (B*n_agents, 2*L) → (B, n_agents, 2*L)

            # ── Path 2: fe + enc — temporal dispatch ─────────────────────
            if model_class_name == "Conv1DModel":
                # (B, n_agents, T, *(obs_shape)) → (B*n_agents*T, *(obs_shape))
                obs_vec = obs.reshape(bs * n_agents * ts, *obs_shape).to(self.device)
                obs_vectorized = fe(obs_vec)
                #   (B*n_agents*T, *(obs_shape)) → (B*n_agents*T, F)
                obs_vectorized = obs_vectorized.reshape(bs * n_agents, ts, -1)
                #   (B*n_agents*T, F) → (B*n_agents, T, F)
                obs_vectorized = obs_vectorized.permute(0, 2, 1)
                #   (B*n_agents, T, F) → (B*n_agents, F, T)  (Conv1D expects channels first)
                local_obs_selected = enc(obs_vectorized)
                #   (B*n_agents, F, T) → (B*n_agents, F)

            elif model_class_name == "RNNModel":
                obs_vec = obs.reshape(bs * n_agents * ts, *obs_shape).to(self.device)
                obs_vectorized = fe(obs_vec)
                #   (B*n_agents*T, *(obs_shape)) → (B*n_agents*T, F)
                obs_vectorized = obs_vectorized.reshape(bs * n_agents, ts, -1)
                #   (B*n_agents*T, F) → (B*n_agents, T, F)
                enc.train()  # cudnn RNN backward only in training mode
                local_obs_selected = enc(obs_vectorized)
                #   (B*n_agents, T, F) → (B*n_agents, F)

            elif model_class_name == "AttentionModel":
                obs_vec = obs.reshape(bs * n_agents * ts, *obs_shape).to(self.device)
                obs_vectorized = fe(obs_vec)
                #   (B*n_agents*T, *(obs_shape)) → (B*n_agents*T, F)
                obs_vectorized = obs_vectorized.reshape(bs * n_agents, ts, -1)
                #   (B*n_agents*T, F) → (B*n_agents, T, F)
                mask = traj_padding_mask[:, idx]
                mask = mask.reshape(bs * n_agents, ts)
                #   (B, n_agents, T) → (B*n_agents, T)
                local_obs_selected = enc(obs_vectorized, mask)
                #   (B*n_agents, T, F) → (B*n_agents, F)

            else:  # MLP / non-temporal — last timestep only
                obs = obs[:, :, -1, :]
                #   (B, n_agents, T, *(obs_shape)) → (B, n_agents, *(obs_shape))
                obs_vec = obs.reshape(bs * n_agents, *obs_shape).to(self.device)
                obs_vectorized = fe(obs_vec)
                #   (B*n_agents, *(obs_shape)) → (B*n_agents, F)
                local_obs_selected = enc(obs_vectorized)
                #   (B*n_agents, F) → (B*n_agents, F)

            local_obs_selected = local_obs_selected.reshape(bs, n_agents, -1)
            #   (B*n_agents, F) → (B, n_agents, F)

            # ── Scatter back to agent-indexed lists ──────────────────────
            agent_latent_selected = agent_latent_selected.permute(1, 0, 2)
            #   (B, n_agents, 2*L) → (n_agents, B, 2*L)
            local_obs_selected = local_obs_selected.permute(1, 0, 2)
            #   (B, n_agents, F) → (n_agents, B, F)

            for j, agent_idx in enumerate(idx):
                agent_latent[agent_idx] = agent_latent_selected[j]
                local_obs[agent_idx]    = local_obs_selected[j]

        agent_latent = torch.stack(agent_latent, dim=1).to(self.device)
        #   (N, B, 2*L) → (B, N, 2*L)
        local_obs = torch.stack(local_obs, dim=1).to(self.device)
        #   (N, B, F) → (B, N, F)

        return agent_latent, local_obs

    def _merge_group_distributions(self, agent_mu, agent_log_var, group_indices):
        if group_indices.max() < 0:
            logging.warning(
                "GroupConsensus: all group_indices are -1 "
                "(no agents grouped — check agent_presence_dim config). "
                "Returning zero-filled group_mu / group_log_var."
            )
            bs, n_agents, f_z = agent_mu.shape
            return (
                torch.zeros(bs, 1, f_z, device=self.device, dtype=agent_mu.dtype),
                torch.zeros(bs, 1, f_z, device=self.device, dtype=agent_log_var.dtype),
            )
        if self.merge_mode == "sample_mean":
            return GroupConsensusAgentGroup._merge_sample_mean(
                self, agent_mu, agent_log_var, group_indices
            )
        return GroupConsensusAgentGroup._merge_bayesian(
            self, agent_mu, agent_log_var, group_indices
        )

    def _merge_sample_mean(self, agent_mu, agent_log_var, group_indices):
        bs, n_agents, f_z = agent_mu.shape
        G = int(group_indices.max()) + 1

        gids = group_indices.to(dtype=torch.long, device=self.device)
        dead = gids < 0
        gids_safe = gids.clamp(min=0)

        mask = F.one_hot(gids_safe, num_classes=G).float()
        mask[dead] = 0.0

        count = mask.sum(dim=1)
        count_safe = count.clamp(min=1)

        group_mu = torch.bmm(mask.transpose(1, 2), agent_mu) / count_safe.unsqueeze(-1)

        neg_inf = torch.tensor(float('-inf'), dtype=agent_log_var.dtype, device=self.device)
        agent_lv_exp = agent_log_var.unsqueeze(2)
        mask_z = mask.unsqueeze(-1)

        max_lv, _ = torch.where(mask_z.bool(), agent_lv_exp, neg_inf).max(dim=1)
        count_mask = (count > 0).unsqueeze(-1)
        max_lv = torch.where(count_mask, max_lv, torch.zeros_like(max_lv))

        log_var_diff = agent_lv_exp - max_lv.unsqueeze(1)
        sum_exp = (torch.exp(log_var_diff) * mask_z).sum(dim=1)
        log_sum_exp = max_lv + torch.log(sum_exp + 1e-8)
        group_log_var = log_sum_exp - 2 * torch.log(count_safe.unsqueeze(-1))
        group_log_var = group_log_var * count_mask
        group_mu = group_mu * count_mask

        return group_mu, group_log_var

    def _merge_bayesian(self, agent_mu, agent_log_var, group_indices):
        bs, n_agents, f_z = agent_mu.shape
        G = int(group_indices.max()) + 1

        gids = group_indices.to(dtype=torch.long, device=self.device)
        dead = gids < 0
        gids_safe = gids.clamp(min=0)

        mask = F.one_hot(gids_safe, num_classes=G).float()
        mask[dead] = 0.0

        count = mask.sum(dim=1)
        count_mask = (count > 0).unsqueeze(-1)

        neg_log_var = -agent_log_var
        neg_inf = torch.tensor(float('-inf'), dtype=agent_log_var.dtype, device=self.device)
        agent_nlv_exp = neg_log_var.unsqueeze(2)
        agent_mu_exp = agent_mu.unsqueeze(2)
        mask_z = mask.unsqueeze(-1)

        max_nlv, _ = torch.where(mask_z.bool(), agent_nlv_exp, neg_inf).max(dim=1)
        max_nlv = torch.where(count_mask, max_nlv, torch.zeros_like(max_nlv))

        nlv_diff = agent_nlv_exp - max_nlv.unsqueeze(1)
        precision = torch.exp(nlv_diff) * mask_z

        sum_precision = precision.sum(dim=1)

        weighted_mu = agent_mu_exp * precision
        sum_weighted_mu = weighted_mu.sum(dim=1)
        group_mu = sum_weighted_mu / sum_precision.clamp(min=1e-8)

        log_sum_precision = max_nlv + torch.log(sum_precision + 1e-8)
        group_log_var = -log_sum_precision

        group_mu = group_mu * count_mask.float()
        group_log_var = group_log_var * count_mask.float()

        return group_mu, group_log_var

    def _merge_group_mean(self, agent_vectors, group_indices):
        bs, n_agents, f_z = agent_vectors.shape
        G = int(group_indices.max()) + 1

        gids = group_indices.to(dtype=torch.long, device=self.device)
        dead = gids < 0
        gids_safe = gids.clamp(min=0)

        mask = F.one_hot(gids_safe, num_classes=G).float()
        mask[dead] = 0.0

        count = mask.sum(dim=1)
        count_safe = count.clamp(min=1)
        count_mask = (count > 0).unsqueeze(-1)

        group_mean = torch.bmm(mask.transpose(1, 2), agent_vectors) / count_safe.unsqueeze(-1)
        group_mean = group_mean * count_mask

        return group_mean, torch.zeros_like(group_mean)

    @staticmethod
    def _scatter(g_t, group_indices):
        bs, G = g_t.shape[:2]
        n_agents = group_indices.shape[1]
        extra_dims = g_t.shape[2:]
        device = g_t.device

        gids = torch.as_tensor(group_indices, dtype=torch.long, device=device)
        dead = gids < 0
        mask = F.one_hot(gids.clamp(min=0), num_classes=G).float()
        mask[dead] = 0.0

        g_t_flat = g_t.reshape(bs, G, -1)
        out_flat = torch.bmm(mask, g_t_flat)
        out = out_flat.reshape(bs, n_agents, *extra_dims)

        return out

    def _process_decoders(self, hidden_states):
        q_val = [None for _ in range(len(self.agent_model_dict))]
        for model_name, dec in self.decoders.items():
            selected_agents = self.model_to_agents[model_name]
            idx = self.model_to_agent_indices[model_name]
            h = hidden_states[:, idx]
            bs = h.shape[0]
            n_agents = len(selected_agents)
            hidden_size = h.shape[-1]
            h = h.reshape(bs * n_agents, hidden_size)
            q_selected = dec(h)
            q_selected = q_selected.reshape(bs, n_agents, -1)

            for j, agent_idx in enumerate(idx):
                q_val[agent_idx] = q_selected[:, j, :]

        q_val = torch.stack(q_val, dim=1).to(self.device)

        return q_val

    def forward(
        self,
        observations: torch.Tensor,
        states: torch.Tensor,
        traj_padding_mask: torch.Tensor,
        alive_mask: torch.Tensor,
        group_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        agent_latent, local_obs = self._process_observations(
            observations, traj_padding_mask
        )

        if group_indices is None:
            group_indices = self.group_builder(states)
            group_indices = group_indices.to(device=alive_mask.device)
            group_indices[~alive_mask] = -1

        if self.consensus_mode == "ae":
            agent_mu = agent_latent
            agent_log_var = torch.zeros_like(agent_latent)
            group_mu, group_log_var = self._merge_group_mean(
                agent_mu, group_indices
            )
            group_consensus = group_mu
        else:
            dim = agent_latent.size(-1) // 2
            agent_mu = agent_latent[:, :, :dim]
            agent_log_var = agent_latent[:, :, dim:]
            # Clamp log_var to keep KL gradient stable.  exp(10) ≈ 2.2e4 gives
            # a tractable gradient magnitude; exp(20) ≈ 4.9e8 already produces
            # a combined loss ~1e7 even with polyak weights, and the resulting
            # clipped gradient is noise-dominated, leading to training collapse.
            agent_log_var = torch.clamp(agent_log_var, min=-10.0, max=10.0)

            group_mu, group_log_var = self._merge_group_distributions(
                agent_mu, agent_log_var, group_indices
            )

            deterministic = self.deterministic_eval and not self.training
            group_consensus, group_log_var, group_mu, _ = process_probabilistic_output(
                torch.cat([group_mu, group_log_var], dim=-1), deterministic
            )

        # Scatter group-level (B,G,L) → per-agent (B,N,L) for RL path
        consensus_per_agent = self._scatter(group_consensus, group_indices)
        consensus_for_rl = consensus_per_agent
        if not self.enable_rl_grad_to_group_estimate:
            consensus_for_rl = consensus_for_rl.detach()
        combined_features = torch.cat((local_obs, consensus_for_rl), dim=-1)

        q_val = self._process_decoders(combined_features)
        q_val = q_val * alive_mask.unsqueeze(-1)

        return {
            "q_val": q_val,
            "group_mu": group_mu,
            "group_log_var": group_log_var,
            "group_consensus": group_consensus,
            "agent_mu": agent_mu,
            "agent_log_var": agent_log_var,
            "group_indices": group_indices,
        }

    def act(
        self,
        observations: Dict[str, np.ndarray],
        state: np.ndarray,
        avail_actions: Dict[str, Any],
        traj_padding_mask: np.ndarray,
        alive_agents: List[str],
        epsilon: float = 0.0,
    ) -> Dict[str, Any]:
        obs = [observations[agent] for agent in self.agent_model_dict.keys()]
        obs = np.stack(obs)
        obs = torch.tensor(obs).unsqueeze(0).to(dtype=torch.float, device=self.device)

        padding_mask = torch.tensor(traj_padding_mask, dtype=torch.bool)
        padding_mask = torch.stack(
            [padding_mask] * len(self.agent_model_dict), dim=0
        )
        padding_mask = padding_mask.unsqueeze(0).to(self.device)

        alive_mask = torch.tensor(
            [agent in set(alive_agents) for agent in self.agent_model_dict.keys()]
        )
        alive_mask = alive_mask.unsqueeze(0).to(self.device)

        with torch.no_grad():
            states_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device=self.device)
            ret = self(
                obs, states_tensor, padding_mask, alive_mask
            )
            q_values = ret["q_val"]
            q_values = q_values.detach().cpu().numpy().squeeze()
            group_indices_arr = ret["group_indices"].squeeze(0)

        if isinstance(next(iter(avail_actions.values())), np.ndarray):
            action_masks = np.array(
                [avail_actions[agent_id] for agent_id in self.agent_model_dict.keys()]
            )
            masked_q_values = np.where(action_masks == 1, q_values, -np.inf)
            optimal_actions = np.argmax(masked_q_values, axis=-1).astype(np.int64)
            mask_probs = action_masks / np.sum(action_masks, axis=1, keepdims=True)
            random_actions = np.array(
                [np.random.choice(len(probs), p=probs) for probs in mask_probs]
            ).astype(np.int64)
        else:
            optimal_actions = np.argmax(q_values, axis=-1).astype(np.int64)
            random_actions = np.array(
                [
                    avail_actions[agent].sample()
                    for agent in self.agent_model_dict.keys()
                ]
            ).astype(np.int64)

        random_choices = np.random.binomial(
            1, epsilon, len(self.agent_model_dict)
        ).astype(np.int64)
        actions = (
            random_choices * random_actions + (1 - random_choices) * optimal_actions
        )
        actions = actions.astype(np.int64).tolist()

        all_actions = {
            agent: action
            for agent, action in zip(self.agent_model_dict.keys(), actions)
        }
        actual_actions = {agent: all_actions[agent] for agent in alive_agents}

        all_group_indices = {
            agent: int(gid)
            for agent, gid in zip(self.agent_model_dict.keys(), group_indices_arr)
        }
        actual_group_indices = {agent: all_group_indices[agent] for agent in alive_agents}

        return {
            "actions": actual_actions,
            "all_actions": all_actions,
            "group_indices": actual_group_indices,
            "all_group_indices": all_group_indices,
        }

    def reset(self) -> "GroupConsensusAgentGroup":
        self.group_builder.reset()
        return self
