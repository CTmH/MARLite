import torch
import numpy as np
import absl.logging as logging
from tqdm import tqdm

from marlite.trainer.self_supervised_qmix_trainer import SelfSupervisedQMIXTrainer
from marlite.util.trajectory_dataset import (
    TrajectoryDataLoader,
)

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
        **kwargs,
    ):
        if recon_mode not in ("per_agent", "per_group"):
            raise ValueError(f"recon_mode must be 'per_agent' or 'per_group', got '{recon_mode}'")
        self.recon_mode = recon_mode
        self.kl_divergence_weight = kl_divergence_weight
        self.warmup_epochs = warmup_epochs
        super().__init__(
            loss_combination_method=loss_combination_method,
            pit_loss_alpha=pit_loss_alpha,
            **kwargs,
        )

    # ── Training loop (single-GPU only) ──────────────────────────────────

    def learn(self, sample_size, batch_size: int, times: int = 1):
        if not self.use_multi_gpu:
            return self._joint_learn_single_gpu(sample_size, batch_size, times)
        return self._joint_learn_single_gpu(sample_size, batch_size, times)

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

        # ── Pre-compute group_indices from state ─────────────────────────
        # Avoids recomputation inside agent.forward().
        # GroupBuilder takes grid state (B, H, W, C) or (B, C, H, W).
        states_last_np = states[:, -1].detach().cpu().numpy()
        #   (B, T, H,W,C) → (B, H,W,C)  numpy  [or (B, C,H,W)]
        group_indices = self.eval_agent_group.group_builder(states_last_np)
        #   (B, N)  numpy, group ID per agent, -1 = dead

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
        #   GroupBuilder skipped (group_indices provided)
        #   merge → group_mu, group_log_var                    (B,N,L),(B,N,L)
        #   sample group_consensus = μ + ε·σ
        #   local_obs ⊕ group_consensus → encoder → decoder → q_val  (B,N,A)

        q_val          = ret["q_val"]           # (B, N, A)
        group_mu       = ret["group_mu"]        # (B, N, L)   same for agents in same group
        group_log_var  = ret["group_log_var"]   # (B, N, L)
        agent_mu       = ret["agent_mu"]        # (B, N, L)   individual agent μ
        agent_log_var  = ret["agent_log_var"]   # (B, N, L)   individual agent log σ²

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

            next_group_indices = self.target_agent_group.group_builder(
                next_states_last_np
            )
            #   (B, N) numpy

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
            targets = self._build_recon_targets(
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

            # ── 3b. Reparameterisation: sample group consensus ───────────
            group_std = torch.exp(0.5 * group_log_var)   # (B, N, L)
            consensus_sample = group_mu + torch.randn_like(group_std) * group_std
            #   (B, N, L) — same consensus for all agents in same group

            # ── 3c. Reconstruction loss ──────────────────────────────────
            if self.recon_mode == "per_group":
                reconstruction_loss = self._recon_loss_per_group(
                    consensus_sample,   # (B, N, L)
                    group_indices,      # (B, N) numpy
                    targets,            # (B, G, …)
                )
            else:
                reconstruction_loss = self._recon_loss_per_agent(
                    consensus_sample,   # (B, N, L)
                    group_indices,      # (B, N) numpy
                    targets,            # (B, G, …)
                )
            #   scalar

            # ── 3d. KL divergence: KL(N(μ_i, σ²_i) || N(0, 1)) ──────────
            # Sum over latent dim, mean over (batch, agent).
            kl_divergence = -0.5 * torch.sum(
                1 + agent_log_var - agent_mu.pow(2) - torch.exp(agent_log_var),
                dim=-1,
            )
            #   (B, N, L) → sum over L → (B, N)
            kl_divergence = torch.mean(kl_divergence)   # (B, N) → scalar

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
            targets:  (B, G, …)  torch float32  [on train_device]
              Shape depends on constructor type.
        """
        obs_np = observations.detach().cpu().numpy()
        #   (B, T, N, obs_dim)  numpy
        states_np = states.detach().cpu().numpy()
        #   (B, T, H,W,C)  numpy
        alive_np = alive_mask.detach().cpu().numpy()
        #   (B, T, N)  numpy bool

        targets_np = self.data_constructor.process(
            observations=obs_np,            # (B, T, N, obs_dim)
            states=states_np,               # (B, T, H,W,C)
            grouping=group_indices,         # (B, N) numpy (pre-computed)
            alive_mask=alive_np,            # (B, T, N) numpy bool
        )
        #   (B, G, …)  numpy float16

        return torch.tensor(targets_np, dtype=torch.float32, device=self.train_device)
        #   (B, G, …)  torch float32

    # ── Reconstruction loss: per-agent mode ──────────────────────────────

    def _recon_loss_per_agent(self, consensus, group_indices, targets):
        """Reconstruction loss computed per agent.

        Each agent independently predicts the group's reconstruction target
        from the shared consensus. Groups contribute proportionally to
        their size (via per-agent averaging of MSE).

        Args:
            consensus:     (B, N, L)  sampled group consensus
            group_indices: (B, N)     numpy group IDs, -1=dead
            targets:       (B, G, …)  reconstruction target

        Returns:
            scalar loss (averaged over active agents).
        """
        bs, n_agents, L = consensus.shape
        device = consensus.device
        G = targets.shape[1]  # n_groups (max)

        total_loss = torch.tensor(0.0, device=device)
        active_agents = 0

        for b in range(bs):
            grp = group_indices[b]                      # (N,) numpy
            unique = torch.unique(grp)                  # sorted unique group IDs
            unique = unique[unique >= 0]                 # filter dead (-1)

            for g in unique:
                if g >= G:
                    continue
                mask = (grp == g)                        # (N,) bool
                n_g = mask.sum().item()                  # agents in group g

                c_per_agent = consensus[b, mask]          # (n_g, L)
                target_g = targets[b, g]                  # (K,K,C_sel) or (F,)
                # Broadcast single group target to all agents
                target_g_expanded = target_g.unsqueeze(0).expand(n_g, *target_g.shape)
                #   (1, …) → (n_g, …)

                pred = self.ssl_model(c_per_agent)        # (n_g, L) → (n_g, …)
                total_loss = total_loss + torch.nn.functional.mse_loss(
                    pred, target_g_expanded, reduction='sum'
                )
                active_agents += n_g

        if active_agents > 0:
            total_loss = total_loss / active_agents
        return total_loss

    # ── Reconstruction loss: per-group mode ──────────────────────────────

    def _recon_loss_per_group(self, consensus, group_indices, targets):
        """Reconstruction loss computed per unique group.

        Deduplicates consensus per group (first agent as representative).
        Equal weight per group regardless of size.

        Args:
            consensus:     (B, N, L)  sampled group consensus
            group_indices: (B, N)     numpy group IDs, -1=dead
            targets:       (B, G, …)  reconstruction target

        Returns:
            scalar loss (averaged over active groups).
        """
        bs, n_agents, L = consensus.shape
        device = consensus.device
        G = targets.shape[1]

        total_loss = torch.tensor(0.0, device=device)
        active_groups = 0

        for b in range(bs):
            grp = group_indices[b]
            unique = torch.unique(grp)
            unique = unique[unique >= 0]

            for g in unique:
                if g >= G:
                    continue
                mask = (grp == g)
                c_g = consensus[b, mask][0]              # (L,) — representative
                pred = self.ssl_model(c_g.unsqueeze(0))   # (1, L) → (1, …)
                target_g = targets[b, g].unsqueeze(0)     # (…) → (1, …)
                total_loss = total_loss + torch.nn.functional.mse_loss(pred, target_g)
                active_groups += 1

        if active_groups > 0:
            total_loss = total_loss / active_groups
        return total_loss
