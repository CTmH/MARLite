"""SSL Group Consensus MAPPO trainer.

Combines the on-policy MAPPO algorithm (PPO clipped surrogate, GAE,
centralised value critic) with SSL-based group consensus learning.
The consensus latent is trained via self-supervised reconstruction of
global state, while the PPO objectives drive the policy and value updates.

SSL infrastructure (model, optimiser, data constructor, checkpoint)
is provided by :class:`SelfSupervisedMAPPOTrainer`.  This class only
adds SSL-specific logic (KL divergence, reconstruction modes, warmup).
"""

import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm
from absl import logging

from marlite.trainer.self_supervised_mappo_trainer import SelfSupervisedMAPPOTrainer
from marlite.trainer.trainer_worker_group.ssl_gc_mappo_worker_group import (
    SSLGroupConsensusMAPPOWorkerGroup,
)
from marlite.util.serialization import get_state_dict, load_state_dict_into
from marlite.util.trajectory_dataset import (
    TrajectoryDataLoader,
    GroupSSLEnrichedTrajectoryDataset,
)


class SSLGroupConsensusMAPPOTrainer(SelfSupervisedMAPPOTrainer):
    """MAPPO trainer with SSL group consensus self-supervised learning.

    Extends :class:`SelfSupervisedMAPPOTrainer` with:
    - KL divergence regularisation on the latent distributions.
    - Per-agent or per-group reconstruction modes.
    - Optional warmup phase (PPO-only before enabling SSL).

    Parameters
    ----------
    kl_divergence_weight : float
        Beta_KL weight for KL divergence term.
    recon_mode : str
        ``"per_agent"`` or ``"per_group"`` reconstruction mode.
    kl_on_agent : bool
        Apply KL divergence on per-agent latent distributions.
    kl_on_group : bool
        Apply KL divergence on per-group (deduplicated) distributions.
    warmup_iterations : int
        Number of initial iterations with PPO-only (no SSL).
    **kwargs
        Forwarded to :class:`SelfSupervisedMAPPOTrainer`.
    """

    def __init__(
        self,
        kl_divergence_weight: float = 0.005,
        recon_mode: str = "per_agent",
        kl_on_agent: bool = True,
        kl_on_group: bool = False,
        warmup_iterations: int = 0,
        consensus_mode: str = "vae",
        **kwargs,
    ):
        if recon_mode not in ("per_agent", "per_group"):
            raise ValueError(
                f"recon_mode must be 'per_agent' or 'per_group', got '{recon_mode}'"
            )
        self.recon_mode = recon_mode
        self.kl_divergence_weight = kl_divergence_weight
        self.kl_on_agent = kl_on_agent
        self.kl_on_group = kl_on_group
        self.warmup_iterations = warmup_iterations
        self.consensus_mode = consensus_mode

        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # SSL reconstruction helpers
    # ------------------------------------------------------------------

    def _recon_loss_per_group(self, consensus, targets, construct_mask):
        bs, G, L = consensus.shape
        target_flat_dim = targets.shape[2:]
        pred_flat = self.ssl_model(consensus.reshape(bs * G, L))
        pred_g = pred_flat.reshape(bs, G, *target_flat_dim)
        return self._compute_ssl_loss(pred_g, targets, construct_mask)

    def _recon_loss_per_agent(
        self, consensus, group_indices, targets, construct_mask, alive_mask
    ):
        bs, G, L = consensus.shape
        n_agents = group_indices.shape[1]
        target_flat_dim = targets.shape[2:]

        pred_flat = self.ssl_model(consensus.reshape(bs * G, L))
        pred_g = pred_flat.reshape(bs, G, *target_flat_dim)

        gids = torch.as_tensor(group_indices, device=consensus.device)
        dead = gids < 0
        mask_oh = F.one_hot(gids.clamp(min=0), num_classes=G).float()
        mask_oh[dead] = 0.0
        mask_oh = mask_oh * construct_mask.unsqueeze(1).float()

        flat_pred = pred_g.reshape(bs, G, -1)
        flat_target = targets.reshape(bs, G, -1)
        pred_agent = torch.bmm(mask_oh, flat_pred).reshape(
            bs, n_agents, *target_flat_dim
        )
        target_agent = torch.bmm(mask_oh, flat_target).reshape(
            bs, n_agents, *target_flat_dim
        )
        agent_alive = alive_mask[:, -1, :]
        return self._compute_ssl_loss(pred_agent, target_agent, agent_alive)

    def _compute_kl_divergence(
        self, agent_mu, agent_log_var, alive_mask, group_mu, group_log_var, construct_mask
    ):
        if self.consensus_mode == "ae":
            return torch.tensor(0.0, device=self.train_device)
        kl = torch.tensor(0.0, device=self.train_device)
        if self.kl_on_agent:
            mask = alive_mask[:, -1, :].unsqueeze(-1).expand_as(agent_mu)
            kl_per_dim = (
                1 + agent_log_var - agent_mu.pow(2) - torch.exp(agent_log_var)
            )
            kl = kl + (-0.5 * (kl_per_dim * mask).sum() / mask.sum().clamp(min=1))
        if self.kl_on_group:
            mask = construct_mask.unsqueeze(-1).expand_as(group_mu)
            kl_per_dim = (
                1 + group_log_var - group_mu.pow(2) - torch.exp(group_log_var)
            )
            kl = kl + (-0.5 * (kl_per_dim * mask).sum() / mask.sum().clamp(min=1))
        return kl

    # ------------------------------------------------------------------
    # Multi-GPU support
    # ------------------------------------------------------------------

    def _create_worker_group(self):
        if not self.use_multi_gpu:
            return None
        return SSLGroupConsensusMAPPOWorkerGroup(
            device_ids=self._get_device_ids(),
            agent_group_config=self.agent_group_config,
            critic_config=self.critic_config,
            critic_optimizer_config=self.critic_optimizer_config,
            agent_optimizer_config=self.agent_optimizer_config,
            gamma=self.gamma,
            clip_epsilon=self.clip_epsilon,
            gae_lambda=self.gae_lambda,
            entropy_coef=self.entropy_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
            ssl_model_config=self.ssl_model_config,
            ssl_optimizer_config=self.ssl_optimizer_config,
            reconstruction_loss=self.reconstruction_loss,
            data_constructor=self.data_constructor,
            self_supervised_learning_loss_weight=self.self_supervised_learning_loss_weight,
            loss_combination_method=self.loss_combination_method,
            pit_loss_alpha=self.pit_loss_alpha,
            kl_divergence_weight=self.kl_divergence_weight,
            recon_mode=self.recon_mode,
            kl_on_agent=self.kl_on_agent,
            kl_on_group=self.kl_on_group,
            warmup_iterations=self.warmup_iterations,
            consensus_mode=self.consensus_mode,
        )

    # ------------------------------------------------------------------
    # PPO + SSL learning (multi-GPU)
    # ------------------------------------------------------------------

    def _learn_multi_gpu(self, sample_size, batch_size: int, times: int = 1):
        """Multi-GPU PPO + SSL learning via worker processes.

        Reconstruction targets are pre-generated **once** on the trainer
        via ``GroupSSLEnrichedTrajectoryDataset`` so workers read
        ``formatted_obs`` / ``construct_padding_mask`` from the batch.
        """
        self.worker_group.move_models_to_gpu()
        total_combined = 0.0
        total_critic = 0.0
        total_ssl = 0.0
        total_batches = 0

        is_warmup = self.current_epoch < self.warmup_iterations

        for epoch in range(times):
            dataset = self.replaybuffer.sample(sample_size)

            if not is_warmup:
                t0 = time.time()
                ssl_dataset = GroupSSLEnrichedTrajectoryDataset(
                    dataset, self.data_constructor
                )
                logging.info(
                    f"  SSL enrichment done in {time.time() - t0:.2f}s "
                    f"({len(dataset)} samples)"
                )
            else:
                ssl_dataset = dataset

            dataloader = TrajectoryDataLoader(
                ssl_dataset, batch_size=batch_size, shuffle=True,
                num_workers=self.n_workers,
            )
            with tqdm(
                total=sample_size, desc=f"Times {epoch + 1}/{times}", unit="batch"
            ) as pbar:
                for batch in dataloader:
                    batch["epoch"] = self.current_epoch
                    result = self.worker_group.train_step(batch)
                    if isinstance(result, tuple):
                        combined, critic, ssl = result
                        total_combined += combined
                        total_critic += critic
                        total_ssl += ssl
                    else:
                        total_combined += result
                    total_batches += 1

                    bs = batch["states"].shape[0]
                    pbar.update(bs)

            self.worker_group.move_models_to_cpu()
        torch.cuda.empty_cache()
        avg_rl = total_critic / max(total_batches, 1)
        avg_ssl = total_ssl / max(total_batches, 1)
        logging.info(f"  Iter {self.current_epoch}: RL Loss {avg_rl:.4f}, SSL Loss {avg_ssl:.4f}")
        return total_combined / max(total_batches, 1)

    # ------------------------------------------------------------------
    # PPO + SSL learning (single-GPU)
    # ------------------------------------------------------------------

    def _learn_single_gpu(self, sample_size, batch_size: int, times: int = 4):
        """Single-GPU PPO + SSL joint learning loop.

        Reconstruction targets are pre-generated **once** for the entire
        sampled dataset via ``GroupSSLEnrichedTrajectoryDataset``.  The
        pre-generated ``formatted_obs`` / ``construct_padding_mask`` are
        read directly from each batch.
        """
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_ssl_loss = 0.0
        total_batches = 0

        self.eval_agent_group.to(self.train_device)
        self.eval_critic.to(self.train_device)
        self.ssl_model.to(self.train_device)

        is_warmup = self.current_epoch < self.warmup_iterations

        dataset = self.replaybuffer.sample(sample_size)

        # ── Pre-generate all reconstruction targets once ──
        if not is_warmup:
            t0 = time.time()
            ssl_dataset = GroupSSLEnrichedTrajectoryDataset(
                dataset, self.data_constructor
            )
            logging.info(
                f"  SSL enrichment done in {time.time() - t0:.2f}s "
                f"({len(dataset)} samples)"
            )
        else:
            ssl_dataset = dataset

        for epoch in range(times):
            dataloader = TrajectoryDataLoader(
                ssl_dataset, batch_size=batch_size, shuffle=True,
                num_workers=self.n_workers,
            )
            with tqdm(
                total=sample_size, desc=f"Times {epoch + 1}/{times}", unit="batch"
            ) as pbar:
                for batch in dataloader:
                    alive_mask = batch["alive_mask"].to(dtype=torch.bool)
                    observations = batch["observations"].to(dtype=torch.float32)
                    timestep_padding_mask = batch["timestep_padding_mask"].to(
                        dtype=torch.bool, device=self.train_device
                    )
                    states = batch["states"].to(dtype=torch.float32)
                    actions = batch["actions"].to(dtype=torch.int)
                    rewards = batch["rewards"].to(dtype=torch.float32)
                    next_states = batch["next_states"].to(dtype=torch.float32)
                    next_timestep_padding_mask = batch[
                        "next_timestep_padding_mask"
                    ].to(dtype=torch.bool, device=self.train_device)
                    next_alive_mask = batch["next_alive_mask"].to(dtype=torch.bool)
                    all_log_probs = batch["all_log_probs"].to(dtype=torch.float32)
                    terminations = batch["terminations"].to(dtype=torch.bool)

                    bs = states.shape[0]
                    n_agents = rewards.shape[2]
                    device = self.train_device

                    alive_mask_d = alive_mask.to(device)
                    next_alive_mask_d = next_alive_mask.to(device)
                    states_dev = states.to(device)
                    next_states_dev = next_states.to(device)

                    # ── Precompute group_indices from batch ──
                    group_indices_batch = batch.get("group_indices")
                    if group_indices_batch is not None:
                        group_indices_np = group_indices_batch[:, -1, :].numpy()
                    else:
                        group_indices_np = np.zeros((bs, n_agents), dtype=np.int64) - 1

                    # ── Critic forward: full sequence → V(s_{T-1}) only ──
                    self.eval_critic.train()
                    v = self.eval_critic(states_dev, alive_mask_d, timestep_padding_mask)["v"]
                    v_last = v[:, 0]

                    with torch.no_grad():
                        v_next = self.eval_critic(
                            next_states_dev[:, -1:, ...],
                            next_alive_mask_d[:, -1:, ...],
                            next_timestep_padding_mask[:, -1:],
                        )["v"][:, 0]

                    # ── Single-step TD residual as advantage ──
                    r_last = self._aggregate_rewards(rewards[:, -1]).to(device)
                    termination_last = terminations[:, -1].prod(dim=-1).to(
                        dtype=torch.float32, device=device
                    )
                    delta = r_last + self.gamma * v_next * (1.0 - termination_last) - v_last
                    advantages_last = delta
                    returns = delta + v_last

                    # ── Agent forward: action_logits + consensus ──
                    states_last = states_dev[:, -1]
                    timestep_padding_mask_expanded = torch.stack(
                        [timestep_padding_mask] * n_agents, dim=1
                    ).to(device)
                    observations_transposed = torch.transpose(observations, 1, 2).to(device)

                    self.eval_agent_group.reset().train()
                    ret_agent = self.eval_agent_group(
                        observations_transposed, states_last,
                        timestep_padding_mask_expanded, alive_mask_d[:, -1, :],
                        group_indices_np,
                    )
                    action_logits = ret_agent["action_logits"]
                    group_consensus = ret_agent.get("group_consensus")
                    group_mu = ret_agent.get("group_mu")
                    group_log_var = ret_agent.get("group_log_var")
                    agent_mu = ret_agent.get("agent_mu")
                    agent_log_var = ret_agent.get("agent_log_var")

                    # ── PPO actor loss ──
                    actions_last = actions[:, -1].to(dtype=torch.int64, device=device)
                    log_probs_old = all_log_probs[:, -1, :].to(device)

                    dist = Categorical(logits=action_logits)
                    new_log_probs = dist.log_prob(actions_last)
                    entropy = dist.entropy()

                    alive_last_flag = alive_mask_d[:, -1, :].to(dtype=torch.float32, device=device)
                    alive_last_count = alive_last_flag.sum()

                    ratio = torch.exp(new_log_probs - log_probs_old)
                    adv_expanded = advantages_last.unsqueeze(-1).expand(-1, n_agents)
                    surr1 = ratio * adv_expanded
                    surr2 = (
                        torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                        * adv_expanded
                    )
                    actor_loss = (
                        -(torch.min(surr1, surr2) * alive_last_flag).sum()
                        / max(alive_last_count, torch.tensor(1.0, device=device))
                    )
                    entropy_loss = (
                        -(entropy * alive_last_flag).sum()
                        / max(alive_last_count, torch.tensor(1.0, device=device))
                    )
                    actor_loss = actor_loss + self.entropy_coef * entropy_loss

                    # ── PPO critic loss ──
                    critic_loss = F.mse_loss(v_last, returns.detach())

                    # ── SSL reconstruction loss ──
                    if is_warmup or group_consensus is None:
                        ssl_loss = torch.tensor(0.0, device=device)
                    else:
                        targets = batch["formatted_obs"].to(
                            dtype=torch.float32, device=device
                        )
                        construct_mask = batch["construct_padding_mask"].to(
                            dtype=torch.bool, device=device
                        )
                        if self.recon_mode == "per_group":
                            recon_loss = self._recon_loss_per_group(
                                group_consensus, targets, construct_mask
                            )
                        else:
                            recon_loss = self._recon_loss_per_agent(
                                group_consensus, group_indices_np, targets,
                                construct_mask, alive_mask_d,
                            )
                        kl = self._compute_kl_divergence(
                            agent_mu, agent_log_var, alive_mask_d,
                            group_mu, group_log_var, construct_mask,
                        )
                        ssl_loss = recon_loss + self.kl_divergence_weight * kl

                    # ── Combined loss ──
                    rl_loss = actor_loss + self.vf_coef * critic_loss
                    if is_warmup or ssl_loss.item() == 0.0:
                        combined_loss = rl_loss
                    else:
                        combined_loss = self._combine_rl_ssl_loss(rl_loss, ssl_loss)

                    # ── Backward ──
                    self.agent_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    self.ssl_optimizer.zero_grad()

                    combined_loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        self.eval_agent_group.parameters(), max_norm=self.max_grad_norm,
                    )
                    torch.nn.utils.clip_grad_norm_(
                        self.eval_critic.parameters(), max_norm=self.max_grad_norm,
                    )
                    torch.nn.utils.clip_grad_norm_(
                        self.ssl_model.parameters(), max_norm=self.max_grad_norm,
                    )

                    self.agent_optimizer.step()
                    self.critic_optimizer.step()
                    self.ssl_optimizer.step()

                    total_actor_loss += actor_loss.detach().cpu().item()
                    total_critic_loss += critic_loss.detach().cpu().item()
                    total_ssl_loss += (
                        ssl_loss.detach().cpu().item()
                        if isinstance(ssl_loss, torch.Tensor) else ssl_loss
                    )
                    total_batches += 1

                    pbar.update(bs)

        self.eval_agent_group.to("cpu")
        self.eval_critic.to("cpu")
        self.ssl_model.to("cpu")
        torch.cuda.empty_cache()

        avg_rl = (total_actor_loss + total_critic_loss * self.vf_coef) / max(
            total_batches, 1
        )
        avg_ssl = total_ssl_loss / max(total_batches, 1)
        logging.info(
            f"  Iter {self.current_epoch}: RL Loss {avg_rl:.4f}, SSL Loss {avg_ssl:.4f}"
        )
        return avg_rl + avg_ssl
