import torch
import torch.nn.functional as F
import numpy as np
import absl.logging as logging
from tqdm import tqdm

from marlite.trainer.self_supervised_qmix_trainer import SelfSupervisedQMIXTrainer
from marlite.trainer.trainer_worker_group.vae_group_consensus_worker_group import (
    VAEGroupConsensusWorkerGroup,
)
from marlite.util.trajectory_dataset import (
    TrajectoryDataLoader,
)
from marlite.util.serialization import get_state_dict, load_state_dict_into

# ────────────────────────────────────────────────────────────────────────
# Dimension reference
# ────────────────────────────────────────────────────────────────────────
# B      : batch size
# T      : trajectory timesteps (last timestep used for state & actions)
# N      : number of agents
# S      : global state dims (H, W, C) or (C, H, W) — grid format
# A      : action space size
# L      : latent_dim  (per-agent latent / group consensus dimension)
# G      : n_groups   (max number of groups)
# K      : window_size (for A2 constructor)
# C_sel  : selected_channels (for A2 constructor)
# F      : N_FEATURES  (for D constructor, = 18)
# ────────────────────────────────────────────────────────────────────────


class VAEGroupConsensusQMIXTrainer(SelfSupervisedQMIXTrainer):
    """GroupConsensus trainer with VAE-style reconstruction loss for consensus.

    Uses a data constructor (MagentGroupWindowConstructor or
    MagentGroupFeaturesConstructor) to generate reconstruction targets from
    the global state, then trains a decoder (ssl_model) to predict these
    targets from the group consensus vectors.

    Key design decisions:
      - Standard QMixer critic (NOT GroupConsensusMixer). Consensus is
        trained solely through reconstruction loss, avoiding nested
        hypernetwork issues. IGM property is fully preserved.
      - Group indices are pre-computed once from state and passed to all
        consumers (agent forward, reconstruction), avoiding recomputation.

    Two reconstruction modes:
      - "per_agent" (recommended): Each agent independently predicts the
        group target from the shared group consensus. Larger groups
        naturally get proportionally more gradient budget.
      - "per_group": Each group predicts its target once, equal weight
        per group regardless of size.

    Loss decomposition:
        L_total = L_TD                                              (always)
                + w_ssl * (L_recon + β_kl * L_KL)                  (after warmup)

    where:
        L_TD    = MSE(Q_tot, r + γ*(1-done)*Q_target)
        L_recon = MSE(ssl_model(c_sample), target_from_state)
        L_KL    = mean_{b,n,d} KL(N(μ_i,σ²_i) || N(0,1))
    """

    def __init__(
        self,
        recon_mode: str = "per_agent",
        kl_divergence_weight: float = 0.005,
        warmup_epochs: int = 0,
        loss_combination_method: str = "weighted_sum",
        pit_loss_alpha: float = 0.9,
        kl_on_group: bool = False,
        kl_on_consensus: bool = True,
        **kwargs,
    ):
        if recon_mode not in ("per_agent", "per_group"):
            raise ValueError(f"recon_mode must be 'per_agent' or 'per_group', got '{recon_mode}'")
        self.recon_mode = recon_mode
        self.kl_divergence_weight = kl_divergence_weight
        self.warmup_epochs = warmup_epochs
        self.kl_on_group = kl_on_group
        self.kl_on_consensus = kl_on_consensus
        super().__init__(
            loss_combination_method=loss_combination_method,
            pit_loss_alpha=pit_loss_alpha,
            **kwargs,
        )

    # ── Multi-GPU support ───────────────────────────────────────────────

    def _create_worker_group(self):
        if not self.use_multi_gpu:
            return None

        return VAEGroupConsensusWorkerGroup(
            device_ids=list(range(len(self.device_list))),
            agent_group_config=self.agent_group_config,
            critic_config=self.critic_config,
            critic_optimizer_config=self.critic_optimizer_config,
            agent_optimizer_config=self.agent_optimizer_config,
            gamma=self.gamma,
            ssl_model_config=self.ssl_model_config,
            ssl_optimizer_config=self.ssl_optimizer_config,
            reconstruction_loss=self.reconstruction_loss,
            data_constructor=self.data_constructor,
            kl_divergence_weight=self.kl_divergence_weight,
            self_supervised_learning_loss_weight=self.self_supervised_learning_loss_weight,
            loss_combination_method=self.loss_combination_method,
            pit_loss_alpha=self.pit_loss_alpha,
            warmup_epochs=self.warmup_epochs,
            recon_mode=self.recon_mode,
            kl_on_group=self.kl_on_group,
            kl_on_consensus=self.kl_on_consensus,
        )

    def _sync_params_to_workers(self):
        if self.worker_group is None:
            return

        trainable_params = {
            "eval_agent_group": get_state_dict(self.eval_agent_group),
            "target_agent_group": get_state_dict(self.target_agent_group),
            "eval_critic": get_state_dict(self.eval_critic),
            "target_critic": get_state_dict(self.target_critic),
        }
        if hasattr(self, "ssl_model") and self.ssl_model is not None:
            trainable_params["ssl_model"] = get_state_dict(self.ssl_model)
        self.worker_group.broadcast_params(trainable_params)

        critic_lr = self.critic_optimizer.param_groups[0]["lr"]
        agent_lr = self.agent_optimizer.param_groups[0]["lr"]
        self.worker_group.sync_lr_to_workers(critic_lr, agent_lr)

    def _sync_eval_params_from_workers(self):
        if self.worker_group is None:
            return
        eval_params = self.worker_group.read_params_from_worker0()
        load_state_dict_into(
            self.eval_agent_group, eval_params["eval_agent_group"]
        )
        load_state_dict_into(self.eval_critic, eval_params["eval_critic"])
        if "ssl_model" in eval_params and hasattr(self, "ssl_model"):
            load_state_dict_into(self.ssl_model, eval_params["ssl_model"])

    # ── Training loop ────────────────────────────────────────────────────

    def learn(self, sample_size, batch_size: int, times: int = 1):
        if not self.use_multi_gpu:
            return self._joint_learn_single_gpu(sample_size, batch_size, times)
        return self._joint_learn_multi_gpu(sample_size, batch_size, times)

    def _joint_learn_single_gpu(self, sample_size, batch_size: int, times: int = 1):
        total_combined = 0.0
        total_critic = 0.0
        total_vae = 0.0
        total_batches = 0

        self.eval_agent_group.to(self.train_device)
        self.eval_critic.to(self.train_device)
        self.target_agent_group.to(self.train_device)
        self.target_critic.to(self.train_device)
        self.ssl_model.to(self.train_device)

        is_warmup = self.current_epoch < self.warmup_epochs

        for t in range(times):
            dataset = self.replaybuffer.sample(sample_size)
            dataloader = TrajectoryDataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.n_workers,
            )
            with tqdm(
                total=sample_size, desc=f"Times {t + 1}/{times}", unit="batch"
            ) as pbar:
                for batch in dataloader:
                    combined_loss, critic_loss, vae_loss = self._compute_loss(
                        batch, is_warmup
                    )

                    self.critic_optimizer.zero_grad()
                    self.agent_optimizer.zero_grad()
                    self.ssl_optimizer.zero_grad()

                    combined_loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        self.eval_critic.parameters(), max_norm=5.0
                    )
                    torch.nn.utils.clip_grad_norm_(
                        self.eval_agent_group.parameters(), max_norm=5.0
                    )
                    torch.nn.utils.clip_grad_norm_(
                        self.ssl_model.parameters(), max_norm=5.0
                    )

                    self.critic_optimizer.step()
                    self.agent_optimizer.step()
                    self.ssl_optimizer.step()

                    total_combined += combined_loss.detach().cpu().item()
                    total_critic += critic_loss.detach().cpu().item()
                    if isinstance(vae_loss, torch.Tensor):
                        total_vae += vae_loss.detach().cpu().item()
                    total_batches += 1

                    bs = batch["states"].shape[0]
                    pbar.update(bs)

        self.eval_agent_group.to("cpu")
        self.eval_critic.to("cpu")
        self.target_agent_group.to("cpu")
        self.target_critic.to("cpu")
        self.ssl_model.to("cpu")
        torch.cuda.empty_cache()

        avg_combined = total_combined / total_batches
        avg_critic = total_critic / total_batches
        avg_vae = total_vae / total_batches
        logging.info(
            f"  Combined Loss: {avg_combined:.4f}, RL Loss: {avg_critic:.4f}, VAE Loss: {avg_vae:.4f}"
        )

        return avg_combined

    def _joint_learn_multi_gpu(self, sample_size, batch_size: int, times: int = 1):
        self.worker_group.move_models_to_gpu()

        total_combined = 0.0
        total_critic = 0.0
        total_vae = 0.0
        total_batches = 0

        for t in range(times):
            dataset = self.replaybuffer.sample(sample_size)
            dataloader = TrajectoryDataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.n_workers,
            )
            with tqdm(
                total=sample_size, desc=f"Times {t + 1}/{times}", unit="batch"
            ) as pbar:
                for batch in dataloader:
                    batch["epoch"] = self.current_epoch
                    combined, critic, vae = self.worker_group.train_step(batch)

                    total_combined += combined
                    total_critic += critic
                    total_vae += vae
                    total_batches += 1

                    bs = batch["states"].shape[0]
                    pbar.update(bs)

        self.worker_group.move_models_to_cpu()
        torch.cuda.empty_cache()

        avg_combined = total_combined / total_batches
        avg_critic = total_critic / total_batches
        avg_vae = total_vae / total_batches
        logging.info(
            f"  Combined Loss: {avg_combined:.4f}, RL Loss: {avg_critic:.4f}, VAE Loss: {avg_vae:.4f}"
        )

        return avg_combined

    # ── Core loss computation ────────────────────────────────────────────
    #
    # NOTE on nn.Module invocation:
    #   All nn.Module instances are called via `module(args)` (i.e. __call__)
    #   rather than `module.forward(args)`.  __call__ invokes registered
    #   forward pre/post hooks, which are essential for:
    #     - DistributedDataParallel (gradient sync hooks)
    #     - torch.compile / torch.jit.trace (capture hooks)
    #     - register_forward_hook / register_full_backward_hook
    #     - Activation checkpointing (checkpoint function hooks)
    #   Direct .forward() bypasses all hooks and should only be used when
    #   hooks must be intentionally skipped.
    # ──────────────────────────────────────────────────────────────────────

    def _compute_loss(self, batch, is_warmup: bool):
        """Compute combined RL+SSL loss in a single forward pass.

        Tensor shape legend:
            B = batch_size    N = n_agents    T = traj timesteps
            L = latent_dim    G = n_groups    A = action_space

        ─── RL path (standard QMixer) ───
            (B,N,T,obs)          → agent group  → q_val       (B,N,A)
            (B,N), (B,T,S_grid)  → standard mixer → q_tot     (B,)
            (B,N,A) max          → q_val_next   (B,N)
            (B,N), (B,T,S_grid)  → target mixer  → q_tot_next (B,)
            L_TD = MSE(q_tot, r + γ·(1-done)·q_tot_next)

        ─── Consensus path (agent only, not critic) ───
            GroupBuilder(state)                  → group_indices  (B,N)
            ge_fe(o_i) → μ_i,logσ²_i             → (B,N,L),(B,N,L)
            merge({μ_i,σ²_i}_{i∈g})              → group_mu       (B,N,L)
                                                  → group_log_var  (B,N,L)

        ─── VAE path (after warmup) ───
            data_constructor(state, grouping)     → targets           (B,G,…)
            c_sample ~ N(group_mu, group_log_var) → consensus_sample  (B,N,L)

            per_agent:
              ssl_model(consensus_sample)          → pred  (B,N,…)
              L_recon = Σ_g Σ_{i∈g} MSE(pred_i, target_g) / Σ_g n_g

            per_group:
              ssl_model(dedup_consensus)           → pred  (Σ_g 1,…)
              L_recon = Σ_g MSE(pred_g, target_g) / n_active_groups

            L_KL = mean_{b,n,d} KL(N(μ,σ²)||N(0,1))

        ─── combined ───
            combined_loss = L_TD + w_ssl·(L_recon + β_kl·L_KL)
        """
        # ── Extract batch tensors ────────────────────────────────────────
        alive_mask = batch["alive_mask"].to(dtype=torch.bool)
        #   (B, T, N)  bool
        observations = batch["observations"].to(dtype=torch.float32)
        #   (B, T, N, obs_dim)  or  (B, T, N, C, H, W)
        timestep_padding_mask = batch["timestep_padding_mask"].to(dtype=torch.bool)
        #   (B, T)
        states = batch["states"].to(dtype=torch.float32)
        #   (B, T, H, W, C)  or  (B, T, C, H, W)  — grid state
        actions = batch["actions"].to(dtype=torch.int)
        #   (B, T, N)
        rewards = batch["rewards"].to(dtype=torch.float32)
        #   (B, T, N)
        next_states = batch["next_states"].to(dtype=torch.float32)
        #   (B, T, H, W, C)  or  (B, T, C, H, W)
        next_observations = batch["next_observations"].to(dtype=torch.float32)
        #   (B, T, N, obs_dim)  or  (B, T, N, C, H, W)
        next_timestep_padding_mask = batch["next_timestep_padding_mask"].to(
            dtype=torch.bool
        )
        #   (B, T)
        next_avail_actions = batch["next_avail_actions"]
        #   (B, T, N, A)  Tensor  or  list of gym spaces
        next_alive_mask = batch["next_alive_mask"].to(dtype=torch.bool)
        #   (B, T, N)
        terminations = batch["terminations"].to(dtype=torch.bool)
        #   (B, T, N)
        bs = states.shape[0]            # B
        n_agents = rewards.shape[2]     # N

        next_alive_mask = next_alive_mask.to(self.train_device)
        alive_mask = alive_mask.to(self.train_device)
        states = states.to(self.train_device)
        next_states = next_states.to(self.train_device)

        if isinstance(next_avail_actions, torch.Tensor):
            use_action_mask = True
            next_avail_actions = next_avail_actions[:, -1, :, :]
            #   (B, -, N, A)  →  (B, N, A)   (last timestep)
            next_avail_actions = next_avail_actions.to(
                dtype=torch.bool, device=self.train_device
            )
            #   (B, N, A)  bool
        else:
            use_action_mask = False

        # ── Rewards & terminations ───────────────────────────────────────
        rewards = rewards[:, -1]               # (B, T, N) → (B, N)
        rewards = rewards.sum(dim=1).to(self.train_device)  # (B, N) → (B,)
        terminations = terminations[:, -1]      # (B, T, N) → (B, N)
        terminations = terminations.prod(dim=1).to(self.train_device)  # (B, N) → (B,)

        # ── Expand padding masks to (B, N, T) ────────────────────────────
        timestep_padding_mask = torch.stack(
            [timestep_padding_mask] * n_agents, dim=1
        ).to(self.train_device)
        #   (B, T) → (B, N, T)
        next_timestep_padding_mask = torch.stack(
            [next_timestep_padding_mask] * n_agents, dim=1
        ).to(self.train_device)
        #   (B, T) → (B, N, T)

        # ── Extract state for agent forward ─────────────────────────────────
        states_last_np = states[:, -1].detach().cpu().numpy()
        #   (B, T, H,W,C) → (B, H,W,C)  numpy

        # ── Pre-compute group_indices from batch ───────────────────────────
        # Stored during rollout, avoids recomputation via group_builder.
        group_indices = batch["group_indices"][:, -1, :].numpy()
        #   (B, T, N) → (B, N)  numpy, group ID per agent, -1 = dead

        # ═══════════════════════════════════════════════════════════════════
        #  PART 1 : RL Forward Pass  (Agent + standard QMixer)
        # ═══════════════════════════════════════════════════════════════════

        self.eval_agent_group.reset().train()
        observations_t = torch.transpose(observations, 1, 2).to(self.train_device)
        #   (B, T, N, obs_dim)  →  (B, N, T, obs_dim)

        # __call__ (not .forward) triggers hooks — essential for DDP / torch.compile
        ret = self.eval_agent_group(
            observations_t,              # (B, N, T, obs_dim)
            states_last_np,              # (B, H,W,C) numpy — passed through
            timestep_padding_mask,       # (B, N, T)
            alive_mask[:, -1, :],        # (B, T, N) → (B, N)
            group_indices,               # (B, N) numpy — pre-computed
        )
        # agent forward internals:
        #   ge_fe(o) → μ,logσ²    → agent_mu, agent_log_var   (B,N,L),(B,N,L)
        #   merge → group_mu, group_log_var (B,N,L), dedup → (B,G,L)
        #   sample group_consensus (B,G,L)
        #   scatter → local_obs ⊕ consensus → decoder → q_val  (B,N,A)

        q_val          = ret["q_val"]           # (B, N, A)
        group_mu       = ret["group_mu"]        # (B, G, L)   deduplicated
        group_log_var  = ret["group_log_var"]   # (B, G, L)
        group_consensus = ret["group_consensus"] # (B, G, L)  already sampled
        agent_mu       = ret["agent_mu"]        # (B, N, L)   individual agent
        agent_log_var  = ret["agent_log_var"]   # (B, N, L)

        # ── Select Q-values at taken actions ─────────────────────────────
        actions_last = actions[:, -1].to(device=self.train_device, dtype=torch.int64)
        #   (B, T, N) → (B, N)
        q_val = torch.gather(q_val, dim=-1, index=actions_last.unsqueeze(-1)).squeeze(-1)
        #   (B, N, A) → (B, N, 1) → (B, N)

        # ── Standard QMixer forward ──────────────────────────────────────
        self.eval_critic.train()
        ret_critic = self.eval_critic(
            q_val,                           # (B, N)
            states,                          # (B, T, H,W,C) — mixer uses last timestep
            alive_mask,                      # (B, T, N)
            timestep_padding_mask[:, 0, :],  # (B, N, T) → pick row 0 → (B, T)
        )
        # Standard QMixer internals:
        #   fe(state_last)     → encoded_states  (B, fe_dim)
        #   QMix model(q_val, state) → q_tot     (B,)
        q_tot = ret_critic["q_tot"]          # (B,)

        # ═══════════════════════════════════════════════════════════════════
        #  PART 2 : TD Targets
        # ═══════════════════════════════════════════════════════════════════

        with torch.no_grad():
            self.target_agent_group.reset().eval()
            next_observations_t = torch.transpose(next_observations, 1, 2).to(
                self.train_device
            )
            #   (B, T, N, obs_dim) → (B, N, T, obs_dim)

            next_states_last_np = next_states[:, -1].detach().cpu().numpy()
            #   (B, T, H,W,C) → (B, H,W,C) numpy

            next_group_indices = batch["next_group_indices"][:, -1, :].numpy()
            #   (B, T, N) → (B, N)  numpy

            ret_next = self.target_agent_group(
                next_observations_t,                 # (B, N, T, obs_dim)
                next_states_last_np,                 # (B, H,W,C) numpy
                next_timestep_padding_mask,          # (B, N, T)
                next_alive_mask[:, -1, :],           # (B, T, N) → (B, N)
                next_group_indices,                  # (B, N) numpy
            )
            q_val_next = ret_next["q_val"]           # (B, N, A)

            # Action masking
            if use_action_mask:
                q_val_next = torch.masked_fill(
                    q_val_next, ~next_avail_actions, -torch.inf
                )
                #   (B, N, A) — unavailable → -inf
            q_val_next = q_val_next.max(dim=-1).values  # (B, N, A) → (B, N)

            self.target_critic.eval()
            ret_next_critic = self.target_critic(
                q_val_next,                              # (B, N)
                next_states,                             # (B, T, H,W,C)
                next_alive_mask,                         # (B, T, N)
                next_timestep_padding_mask[:, 0, :],     # (B, T)
            )
            q_tot_next = ret_next_critic["q_tot"]        # (B,)

        # TD target: y = r + γ·(1 - done)·Q_target(s', argmax Q_target)
        y_tot = rewards + (1 - terminations) * self.gamma * q_tot_next
        #   (B,) + (B,) · scalar · (B,) → (B,)

        critic_loss = torch.nn.functional.mse_loss(q_tot, y_tot.detach())
        #   scalar

        # ═══════════════════════════════════════════════════════════════════
        #  PART 3 : VAE Reconstruction Loss  (after warmup)
        # ═══════════════════════════════════════════════════════════════════

        if is_warmup:
            vae_loss = torch.tensor(0.0, device=self.train_device)
        else:
            # ── 3a. Build reconstruction targets from state & grouping ───
            targets, construct_mask = self._build_recon_targets(
                observations,         # (B, T, N, obs_dim)
                states,               # (B, T, H,W,C)
                group_indices,        # (B, N) numpy (pre-computed)
                alive_mask,           # (B, T, N)
            )
            #   (B, G, …)  shape depends on constructor:
            #     A2 (MagentGroupWindowConstructor):
            #       channel_first=F → (B, G, K, K, C_sel)
            #       channel_first=T → (B, G, C_sel, K, K)
            #     D (MagentGroupFeaturesConstructor):
            #       → (B, G, F)   F = N_FEATURES = 18

            # ── 3b. Group consensus already sampled in agent ────────────
            #   group_consensus: (B, G, L) — sampled latent for each group

            # ── 3c. Reconstruction loss ──────────────────────────────────
            if self.recon_mode == "per_group":
                reconstruction_loss = self._recon_loss_per_group(
                    group_consensus,    # (B, G, L)
                    targets,            # (B, G, …)
                    construct_mask,     # (B, G)
                )
            else:
                reconstruction_loss = self._recon_loss_per_agent(
                    group_consensus,    # (B, G, L)
                    group_indices,      # (B, N) numpy
                    targets,            # (B, G, …)
                    construct_mask,     # (B, G)
                    alive_mask,         # (B, T, N)
                )
            #   scalar

            # ── 3d. KL divergence: KL(N(μ, σ²) || N(0, 1)) ──────────
            kl_divergence = 0.0
            if self.kl_on_consensus:
                kl_mu = agent_mu
                kl_log_var = agent_log_var
                mask = alive_mask[:, -1, :].unsqueeze(-1).expand_as(agent_mu)
                kl_per_dim = 1 + kl_log_var - kl_mu.pow(2) - torch.exp(kl_log_var)
                kl_divergence = kl_divergence + -0.5 * (
                    kl_per_dim * mask
                ).sum() / mask.sum().clamp(min=1)
            if self.kl_on_group:
                kl_mu = group_mu
                kl_log_var = group_log_var
                mask = construct_mask.unsqueeze(-1).expand_as(group_mu)
                kl_per_dim = 1 + kl_log_var - kl_mu.pow(2) - torch.exp(kl_log_var)
                kl_divergence = kl_divergence + -0.5 * (
                    kl_per_dim * mask
                ).sum() / mask.sum().clamp(min=1)

            vae_loss = reconstruction_loss + self.kl_divergence_weight * kl_divergence
            #   scalar

        # ═══════════════════════════════════════════════════════════════════
        #  PART 4 : Combined loss
        # ═══════════════════════════════════════════════════════════════════

        if is_warmup:
            combined_loss = critic_loss
        else:
            combined_loss = self._combine_rl_ssl_loss(critic_loss, vae_loss)
            #   weighted_sum: L_TD + w_ssl * L_VAE
            #   pit_loss:     PITLoss([L_TD, L_VAE])

        return combined_loss, critic_loss, vae_loss

    # ── Reconstruction target builder ────────────────────────────────────

    def _build_recon_targets(self, observations, states, group_indices, alive_mask):
        """Call the data constructor to generate reconstruction targets.

        Args:
            observations:  (B, T, N, obs_dim)  torch float32  [on GPU]
            states:        (B, T, H,W,C)       torch float32  [on GPU]
            group_indices: (B, N)              numpy int8
            alive_mask:    (B, T, N)           torch bool     [on GPU]

        Returns:
            targets:          (B, G, …)  torch float32  [on train_device]
            construct_mask:   (B, G)     torch bool     [on train_device]
        """
        obs_np = observations.detach().cpu().numpy()
        #   (B, T, N, obs_dim)  numpy
        states_np = states.detach().cpu().numpy()
        #   (B, T, H,W,C)  numpy
        alive_np = alive_mask.detach().cpu().numpy()
        #   (B, T, N)  numpy bool

        targets_np, construct_mask_np = self.data_constructor.process(
            observations=obs_np,            # (B, T, N, obs_dim)
            states=states_np,               # (B, T, H,W,C)
            grouping=group_indices,         # (B, N) numpy (pre-computed)
            alive_mask=alive_np,            # (B, T, N) numpy bool
        )
        #   (B, G, …)  numpy float16, (B, G) numpy bool

        targets = torch.tensor(targets_np, dtype=torch.float32, device=self.train_device)
        #   (B, G, …)  torch float32
        construct_mask = torch.tensor(construct_mask_np, dtype=torch.bool, device=self.train_device)
        #   (B, G)  torch bool
        return targets, construct_mask

    # ── Reconstruction loss: per-agent mode ──────────────────────────────

    def _recon_loss_per_agent(self, consensus, group_indices, targets, construct_mask, alive_mask):
        bs, G, L = consensus.shape
        n_agents = group_indices.shape[1]
        device = consensus.device
        target_flat_dim = targets.shape[2:]

        pred_flat = self.ssl_model(consensus.reshape(bs * G, L))
        pred_g = pred_flat.reshape(bs, G, *target_flat_dim)

        gids = torch.as_tensor(group_indices, device=device)
        dead = gids < 0
        mask = F.one_hot(gids.clamp(min=0), num_classes=G).float()
        mask[dead] = 0.0
        mask = mask * construct_mask.unsqueeze(1).float()

        flat_pred = pred_g.reshape(bs, G, -1)
        flat_target = targets.reshape(bs, G, -1)
        pred_agent = torch.bmm(mask, flat_pred).reshape(bs, n_agents, *target_flat_dim)
        target_agent = torch.bmm(mask, flat_target).reshape(bs, n_agents, *target_flat_dim)

        agent_alive = alive_mask[:, -1, :]
        return self._compute_ssl_loss(pred_agent, target_agent, agent_alive)

    # ── Reconstruction loss: per-group mode ──────────────────────────────

    def _recon_loss_per_group(self, consensus, targets, construct_mask):
        """Reconstruction loss per unique group, uniformly weighted.

        Args:
            consensus:      (B, G, L)  already deduplicated group consensus
            targets:        (B, G, …)  reconstruction target
            construct_mask: (B, G)     torch bool — True if group has data
        """
        bs, G, L = consensus.shape
        target_flat_dim = targets.shape[2:]
        D = targets.numel() // (bs * G)

        # ssl_model on group level: (B*G, L) → (B*G, D)
        pred_flat = self.ssl_model(consensus.reshape(bs * G, L))
        pred_g = pred_flat.reshape(bs, G, *target_flat_dim)

        return self._compute_ssl_loss(pred_g, targets, construct_mask)
