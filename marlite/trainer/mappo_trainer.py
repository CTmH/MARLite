"""MAPPO Trainer — Multi-Agent PPO with on-policy training loop.

Implements the MAPPO algorithm (Yu et al., 2021) for discrete action spaces.
Uses Generalized Advantage Estimation (GAE) with a centralized value critic
and per-agent stochastic policies (categorical distributions).  Supports both
single-GPU and multi-GPU training via replicated workers.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm
from absl import logging

from marlite.trainer.onpolicy_trainer import OnPolicyTrainer
from marlite.trainer.trainer_worker_group.mappo_worker_group import MAPPOWorkerGroup
from marlite.algorithm.critic.mixer import Mixer as MixerCritic
from marlite.util.trajectory_dataset import TrajectoryDataLoader


class MAPPOTrainer(OnPolicyTrainer):
    """MAPPO (Multi-Agent Proximal Policy Optimization) trainer.

    On-policy actor-critic algorithm with:
    - Centralized critic V(s) estimating the value of the global state.
    - Per-agent stochastic policy pi(a|o) modelled as a categorical distribution.
    - GAE for computing advantage estimates over trajectory segments.
    - PPO clipped surrogate objective for stable policy updates.

    Supports single-GPU (main process) and multi-GPU (worker processes) modes.
    In on-policy mode, the replay buffer is cleared after each iteration and
    refilled by ``evaluate()`` calls, so each batch of data is used exactly once.

    Parameters
    ----------
    clip_epsilon : float
        PPO clip range for the importance sampling ratio.
    gae_lambda : float
        GAE lambda parameter controlling bias-variance tradeoff.
    entropy_coef : float
        Coefficient for the entropy bonus (encourages exploration).
    vf_coef : float
        Coefficient for the value function loss in the combined loss.
    max_grad_norm : float
        Maximum gradient norm for clipping.
    **kwargs :
        Forwarded to ``OnPolicyTrainer.__init__``.
    """

    def __init__(
        self,
        clip_epsilon: float = 0.2,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 5.0,
        **kwargs,
    ):
        # Must be set before super().__init__() because _create_worker_group()
        # is called during _setup_multi_gpu() in OnPolicyTrainer.__init__().
        self.clip_epsilon = clip_epsilon
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        super().__init__(**kwargs)
        if isinstance(self.eval_critic, MixerCritic):
            raise TypeError(
                "Critic subclass required, not Mixer subclass"
            )

    # ------------------------------------------------------------------
    # Multi-GPU worker group factory
    # ------------------------------------------------------------------

    def _create_worker_group(self):
        """Create a MAPPOWorkerGroup for multi-GPU training.

        Returns ``None`` when ``use_multi_gpu`` is ``False`` (single-GPU mode).
        """
        if not self.use_multi_gpu:
            return None

        return MAPPOWorkerGroup(
            device_ids=list(range(len(self.device_list))),
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
        )

    # ------------------------------------------------------------------
    # PPO learning (single- and multi-GPU)
    # ------------------------------------------------------------------

    def learn(self, sample_size, batch_size: int, times: int = 4):
        """Run PPO updates on data sampled from the replay buffer.

        Parameters
        ----------
        sample_size : int
            Number of transitions to sample from the replay buffer.
        batch_size : int
            Mini-batch size for gradient computation.
        times : int
            Number of PPO epochs per learning call.

        Returns
        -------
        float
            Combined loss (actor + vf_coef * critic) averaged over batches.
        """
        if not self.use_multi_gpu:
            return self._learn_single_gpu(sample_size, batch_size, times)
        return self._learn_multi_gpu(sample_size, batch_size, times)

    def _learn_single_gpu(self, sample_size, batch_size: int, times: int = 4):
        """Single-GPU PPO learning loop.

        For each PPO epoch (``times``), the full sampled dataset is
        iterated over in mini-batches.  Each batch runs:
          - forward pass through the critic (V(s))
          - forward pass through the agent (action logits)
          - TD residual advantage estimation
          - PPO clipped surrogate + entropy bonus  (actor)
          - MSE value loss  (critic)
          - separate backward passes with gradient clipping and optimizer steps.
        """
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_batches = 0

        self.eval_agent_group.to(self.train_device)
        self.eval_critic.to(self.train_device)

        dataset = self.replaybuffer.sample(sample_size)

        for epoch in range(times):
            dataloader = TrajectoryDataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
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
                    next_timestep_padding_mask = batch["next_timestep_padding_mask"].to(
                        dtype=torch.bool, device=self.train_device
                    )
                    next_alive_mask = batch["next_alive_mask"].to(dtype=torch.bool)
                    all_log_probs = batch["all_log_probs"].to(dtype=torch.float32)
                    terminations = batch["terminations"].to(dtype=torch.bool)

                    bs = states.shape[0]
                    n_agents = rewards.shape[2]
                    t_steps = rewards.shape[1]

                    device = self.train_device
                    alive_mask = alive_mask.to(device)
                    next_alive_mask = next_alive_mask.to(device)
                    states_dev = states.to(device)
                    next_states_dev = next_states.to(device)

                    rewards_sum = rewards.sum(dim=2).to(device)
                    terminations_any = terminations.any(dim=2).to(
                        dtype=torch.float32, device=device
                    )

                    timestep_padding_mask_expanded = torch.stack(
                        [timestep_padding_mask] * n_agents, dim=1
                    ).to(device)

                # ---- Critic forward: full sequence -> V(s_{T-1}) only ----
                self.eval_critic.train()
                v = self.eval_critic(states_dev, alive_mask, timestep_padding_mask)["v"]
                v_last = v[:, 0]  # (B,) — value at the last timestep of the segment

                # ---- Bootstrap: V(s_T) from the next state after the segment ----
                with torch.no_grad():
                    v_next = self.eval_critic(
                        next_states_dev[:, -1:, ...],
                        next_alive_mask[:, -1:, ...],
                        next_timestep_padding_mask[:, -1:],
                    )["v"][:, 0]  # (B,)

                # ---- Single-step TD residual as advantage (GAE with one timestep) ----
                r_last = rewards.sum(dim=2)[:, -1].to(device)  # (B,)
                done_last = terminations.any(dim=2)[:, -1].to(
                    dtype=torch.float32, device=device
                )
                delta = r_last + self.gamma * v_next * (1.0 - done_last) - v_last
                advantages_last = delta  # (B,)
                returns = delta + v_last  # (B,) — TD target for the critic

                # ---- Actor forward: action logits for last timestep ----
                observations_transposed = torch.transpose(observations, 1, 2).to(
                    device
                )
                self.eval_agent_group.train()
                ret_agent = self.eval_agent_group(
                    observations_transposed,
                    timestep_padding_mask_expanded,
                    alive_mask[:, -1, :],
                )
                action_logits = ret_agent["action_logits"]  # (B, N, action_dim)

                actions_last = actions[:, -1].to(dtype=torch.int64, device=device)
                log_probs_old = all_log_probs[:, -1, :].to(device)

                # ---- PPO actor loss ----
                dist = Categorical(logits=action_logits)
                new_log_probs = dist.log_prob(actions_last)
                entropy = dist.entropy()

                alive_last_flag = alive_mask[:, -1, :].to(
                    dtype=torch.float32, device=device
                )
                alive_last_count = alive_last_flag.sum()

                ratio = torch.exp(new_log_probs - log_probs_old)
                adv_expanded = advantages_last.unsqueeze(-1).expand(-1, n_agents)  # (B, N)
                surr1 = ratio * adv_expanded
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
                    )
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

                # ---- Critic value loss (single timestep) ----
                critic_loss = F.mse_loss(v_last, returns.detach())

                # ---- Backward pass: actor ----
                self.agent_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(
                    self.eval_agent_group.parameters(), max_norm=self.max_grad_norm
                )

                # ---- Backward pass: critic ----
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.eval_critic.parameters(), max_norm=self.max_grad_norm
                )

                self.agent_optimizer.step()
                self.critic_optimizer.step()

                total_actor_loss += actor_loss.detach().cpu().item()
                total_critic_loss += critic_loss.detach().cpu().item()
                total_batches += 1

                bs = batch["states"].shape[0]
                pbar.update(bs)

        self.eval_agent_group.to("cpu")
        self.eval_critic.to("cpu")
        torch.cuda.empty_cache()

        avg_actor = total_actor_loss / max(total_batches, 1)
        avg_critic = total_critic_loss / max(total_batches, 1)
        return avg_actor + avg_critic * self.vf_coef

    def _learn_multi_gpu(self, sample_size, batch_size: int, times: int = 4):
        """Multi-GPU PPO learning via worker processes.

        Each worker holds a full copy of the eval models and optimizers.
        Batches are sliced across workers, gradients are synchronised via
        all_reduce, and the average combined loss is collected.

        For each PPO epoch (``times``), a progress bar shows batch progress.
        """
        self.worker_group.move_models_to_gpu()

        total_combined = 0.0
        total_batches = 0

        for epoch in range(times):
            dataset = self.replaybuffer.sample(sample_size)
            dataloader = TrajectoryDataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.n_workers,
            )
            with tqdm(
                total=sample_size, desc=f"Times {epoch + 1}/{times}", unit="batch"
            ) as pbar:
                for batch in dataloader:
                    loss = self.worker_group.train_step(batch)
                    total_combined += loss
                    total_batches += 1

                    bs = batch["states"].shape[0]
                    pbar.update(bs)

        self.worker_group.move_models_to_cpu()
        torch.cuda.empty_cache()

        return total_combined / max(total_batches, 1)

    # ------------------------------------------------------------------
    # On-policy training loop
    # ------------------------------------------------------------------

    def train(
        self,
        iterations,
        target_first_metric,
        batch_size=64,
        learning_times_per_iteration=1,
    ):
        """Run the on-policy MAPPO training loop.

        The loop follows this on-policy pattern::

            1. Initial ``evaluate()`` fills the replay buffer with episodes.
            2. For each iteration:
               a. ``learn()`` performs PPO updates on the current buffer.
               b. The buffer is cleared (on-policy: each batch is used once).
               c. ``evaluate()`` generates fresh episodes with the updated
                  policy and adds them to the buffer for the next iteration.
               d. LR schedulers are stepped; best model is tracked.

        Parameters
        ----------
        iterations : int
            Number of training iterations.
        target_first_metric : float
            Target value for the first eval metric (early stopping threshold).
        batch_size : int
            Mini-batch size for PPO updates.
        learning_times_per_iteration : int
            Number of PPO epochs per iteration.

        Returns
        -------
        dict
            Best metrics achieved during training.
        """
        self.eval_episodes_to_replay_ratio = 1.0

        # Initial data collection
        self.evaluate()

        for iteration in range(iterations):
            self.current_epoch = iteration

            sample_size = len(self.replaybuffer.buffer)
            if self.sample_mode == "ratio":
                sample_ratio = self.sample_ratio.get_value(iteration)
                sample_size = round(sample_size * sample_ratio)
            else:
                sample_size = round(self.sample_ratio.get_value(iteration))
            sample_size = min(sample_size, len(self.replaybuffer.buffer))
            if sample_size > 0:
                agent_group_lr = self.agent_optimizer.param_groups[0]["lr"]
                critic_lr = self.critic_optimizer.param_groups[0]["lr"]
                logging.info(
                    f"Iteration {iteration}: Batch size: {batch_size}, "
                    f"Critic lr: {critic_lr:.8f}, Agent lr: {agent_group_lr:.8f}"
                )
                self._sync_params_to_workers()
                loss = self.learn(
                    sample_size=sample_size,
                    batch_size=batch_size,
                    times=learning_times_per_iteration,
                )
                self._sync_eval_params_from_workers()
                logging.info(f"Iteration {iteration}: Loss {loss:.4f}")

            # Clear buffer (on-policy: discard old data)
            self.replaybuffer = self.replaybuffer_config.create_replaybuffer()

            # Collect fresh episodes with updated policy
            result = self.evaluate()
            metrics = {key: result[key]["mean"] for key in self.eval_metric_list}
            first_metric = next(iter(metrics.values()))
            first_metric_name = next(iter(metrics.keys()))
            self.save_intermediate_results(iteration, result)

            if isinstance(
                self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.lr_scheduler.step(first_metric)
            elif isinstance(self.lr_scheduler, torch.optim.lr_scheduler.LRScheduler):
                self.lr_scheduler.step()

            if isinstance(
                self.agent_lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.agent_lr_scheduler.step(first_metric)
            elif isinstance(
                self.agent_lr_scheduler, torch.optim.lr_scheduler.LRScheduler
            ):
                self.agent_lr_scheduler.step()

            if first_metric >= self.best_metrics.get(first_metric_name, -np.inf):
                self.best_metrics = metrics
                self.save_current_model(checkpoint="best")

            if first_metric >= target_first_metric:
                break

        return self.best_metrics
