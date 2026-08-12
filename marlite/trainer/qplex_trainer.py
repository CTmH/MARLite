"""QPLEX Trainer (single-GPU).

Off-policy trainer for the QPLEX algorithm (Wang et al., ICLR 2021).
Mirrors :class:`QMIXTrainer` but uses a :class:`QPLEXMixer` that
consumes the **full** per-agent Q-values (``(B, N, A)``) together with
the per-agent action indices and returns the decomposed ``q_tot``,
``v_tot``, ``a_tot`` and ``att_reg``.

The learn procedure implements Double DQN with the following steps:

1. Eval agent group produces ``(B, N, A)`` Q-values.
2. Eval mixer receives full Q-values + actual actions → ``q_tot``.
3. Eval agent group (eval mode) produces Q-values for the next state;
   the best actions are selected via argmax (Double Q).
4. Target agent group produces next-state Q-values; the target mixer
   receives them with the eval-selected best actions → ``q_tot_next``.
5. TD target: ``y = r + γ * (1 - done) * q_tot_next``.
6. Loss: MSE + optional attention regulariser.
"""

import torch
from tqdm import tqdm

from marlite.trainer.offpolicy_trainer import OffPolicyTrainer
from marlite.util.trajectory_dataset import TrajectoryDataLoader


class QPLEXTrainer(OffPolicyTrainer):
    """Off-policy QPLEX trainer.

    All keyword arguments are forwarded to :class:`OffPolicyTrainer`.
    Since QPLEX does not require an auxiliary ``V_net`` (the joint
    value is produced inside the mixer), no extra constructor
    parameters are needed beyond the standard off-policy set.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _create_worker_group(self):
        """Create multi-GPU worker group (placeholder; not yet implemented)."""
        return None

    def learn(self, sample_size, batch_size: int, times: int = 1):
        """Perform one or more passes of gradient-based learning.

        Delegates to the single- or multi-GPU implementation depending
        on ``self.use_multi_gpu``.
        """
        if not self.use_multi_gpu:
            return self._learn_single_gpu(sample_size, batch_size, times)
        return self._learn_multi_gpu(sample_size, batch_size, times)

    def _learn_single_gpu(self, sample_size, batch_size: int, times: int = 1):
        """Single-GPU QPLEX learning loop.

        Args:
            sample_size: Number of trajectory segments to sample from
                the replay buffer.
            batch_size: Mini-batch size for the data loader.
            times: Number of passes over the sampled data.

        Returns:
            Average total loss (MSE + optional att_reg) over all
            batches in this call.
        """
        total_loss = 0.0
        total_batches = 0

        self.eval_agent_group.to(self.train_device)
        self.eval_critic.to(self.train_device)
        self.target_agent_group.to(self.train_device)
        self.target_critic.to(self.train_device)

        for t in range(times):
            with tqdm(
                total=sample_size, desc=f"Times {t + 1}/{times}", unit="batch"
            ) as pbar:
                dataset = self.replaybuffer.sample(sample_size)
                dataloader = TrajectoryDataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=self.n_workers,
                )
                for batch in dataloader:
                    # ------------------------------------------------------------------
                    # Load batch and move tensors to the compute device.
                    # ------------------------------------------------------------------
                    alive_mask = batch["alive_mask"].to(dtype=torch.bool)
                    observations = batch["observations"].to(dtype=torch.float32)
                    timestep_padding_mask = batch["timestep_padding_mask"].to(
                        dtype=torch.bool
                    )
                    states = batch["states"].to(dtype=torch.float32)
                    actions = batch["actions"].to(dtype=torch.int)
                    rewards = batch["rewards"].to(dtype=torch.float32)
                    next_states = batch["next_states"].to(dtype=torch.float32)
                    next_observations = batch["next_observations"].to(
                        dtype=torch.float32
                    )
                    next_timestep_padding_mask = batch["next_timestep_padding_mask"].to(
                        dtype=torch.bool
                    )
                    next_avail_actions = batch["next_avail_actions"]
                    next_alive_mask = batch["next_alive_mask"].to(dtype=torch.bool)
                    terminations = batch["terminations"].to(dtype=torch.bool)
                    n_agents = rewards.shape[2]

                    next_alive_mask = next_alive_mask.to(self.train_device)
                    alive_mask = alive_mask.to(self.train_device)

                    # ------------------------------------------------------------------
                    # Action mask handling (available actions from the environment).
                    # ------------------------------------------------------------------
                    if isinstance(next_avail_actions, torch.Tensor):
                        use_action_mask = True
                        next_avail_actions = next_avail_actions[:, -1, :, :]
                        next_avail_actions = next_avail_actions.to(
                            dtype=torch.bool, device=self.train_device
                        )
                    else:
                        use_action_mask = False

                    # Rewards and terminations: only the last timestep matters for
                    # 1-step TD; agents share a single joint reward.
                    r_last = self._aggregate_rewards(rewards[:, -1]).to(self.train_device)
                    termination_last = terminations[:, -1].prod(dim=1).to(self.train_device)

                    timestep_padding_mask = torch.stack(
                        [timestep_padding_mask] * n_agents, dim=1
                    ).to(self.train_device)
                    next_timestep_padding_mask = torch.stack(
                        [next_timestep_padding_mask] * n_agents, dim=1
                    ).to(self.train_device)

                    # ------------------------------------------------------------------
                    # Forward: eval networks.
                    #
                    # The agent group computes individual Q-values
                    # Q_i(τ_i, ·)  — one per action per agent.
                    # The QPLEXMixer then factorises these into the
                    # joint Q_tot(τ, a) via the duplex dueling structure
                    # (paper Eq. 11).
                    # ------------------------------------------------------------------
                    self.eval_agent_group.train()
                    observations = torch.transpose(observations, 1, 2).to(
                        self.train_device
                    )
                    ret = self.eval_agent_group(
                        observations, timestep_padding_mask, alive_mask[:, -1, :]
                    )
                    q_val_full = ret["q_val"]  # (B, N, A)
                    actions_last = actions[:, -1].to(
                        device=self.train_device, dtype=torch.int64
                    )
                    states = states.to(self.train_device)

                    self.eval_critic.train()
                    cret = self.eval_critic(
                        q_val_full,
                        states,
                        actions_last,
                        alive_mask,
                        timestep_padding_mask[:, 0, :],
                    )
                    q_tot = cret["q_tot"]
                    att_reg = cret["att_reg"]

                    # ------------------------------------------------------------------
                    # Target (Double DQN).
                    #
                    # We follow the standard Double DQN procedure:
                    #   1. The **eval** agent group selects the best actions
                    #      for the next state (argmax over Q_i_eval).
                    #   2. The **target** agent group evaluates the Q-values
                    #      for those actions  (Q_i_target).
                    #   3. The target mixer computes Q_tot_next from those
                    #      target Q-values and the eval-selected actions.
                    #
                    # TD target (paper Eq. 1):
                    #   y = r + γ · (1 − done) · Q_tot(τ', a*)
                    #   where a* = argmax_a Q_tot_eval(τ', ·)
                    #
                    # Loss: L = (y − Q_tot(τ, a))²  (standard 1-step TD)
                    #       + attend_reg_coef · attention regulariser
                    #         (paper Appendix B).
                    # ------------------------------------------------------------------
                    with torch.no_grad():
                        # Eval agent selects best actions (Double Q).
                        self.eval_agent_group.eval()
                        next_observations_t = torch.transpose(
                            next_observations, 1, 2
                        ).to(self.train_device)
                        ret_next_eval = self.eval_agent_group(
                            next_observations_t,
                            next_timestep_padding_mask,
                            next_alive_mask[:, -1, :],
                        )
                        q_val_next_eval = ret_next_eval["q_val"]
                        if use_action_mask:
                            q_val_next_eval = torch.masked_fill(
                                q_val_next_eval,
                                ~next_avail_actions,
                                -torch.inf,
                            )
                        best_actions = q_val_next_eval.argmax(dim=-1)

                        # Target agent evaluates the next state Q for those actions.
                        self.target_agent_group.eval()
                        ret_next_target = self.target_agent_group(
                            next_observations_t,
                            next_timestep_padding_mask,
                            next_alive_mask[:, -1, :],
                        )
                        q_val_next = ret_next_target["q_val"]  # (B, N, A)
                        next_states = next_states.to(self.train_device)
                        self.target_critic.eval()
                        cret_next = self.target_critic(
                            q_val_next,
                            next_states,
                            best_actions,
                            next_alive_mask,
                            next_timestep_padding_mask[:, 0, :],
                        )
                        q_tot_next = cret_next["q_tot"]

                    # ------------------------------------------------------------------
                    # Loss computation (paper Eq. 1 + attention regulariser).
                    #
                    # 1-step TD target:
                    #   y = r + γ · (1 − done) · Q_tot(τ', a*)
                    #
                    # where Q_tot(τ, a) = V_tot(τ) + A_tot(τ, a) (Eq. 11a).
                    # The loss is:
                    #   L = MSE(y, Q_tot(τ, a)) + λ_reg · Σ_k (logit_k)²
                    # ------------------------------------------------------------------
                    y_tot = r_last + (1 - termination_last) * self.gamma * q_tot_next
                    critic_loss = torch.nn.functional.mse_loss(q_tot, y_tot.detach())

                    if att_reg.item() != 0:
                        total_batch_loss = critic_loss + att_reg
                    else:
                        total_batch_loss = critic_loss

                    # ------------------------------------------------------------------
                    # Backprop.
                    # ------------------------------------------------------------------
                    self.agent_optimizer.zero_grad()
                    self.eval_critic.zero_grad()
                    total_batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.eval_critic.parameters(), max_norm=self.max_grad_norm
                    )
                    torch.nn.utils.clip_grad_norm_(
                        self.eval_agent_group.parameters(),
                        max_norm=self.max_grad_norm,
                    )
                    self.critic_optimizer.step()
                    self.agent_optimizer.step()

                    total_loss += total_batch_loss.detach().cpu().item()
                    total_batches += 1

                    # Per-batch target update (hard / ema)
                    self._total_batches_processed += 1
                    self._update_target_after_batch()

                    pbar.update(actions.shape[0])

        # Move back to CPU to free GPU memory.
        self.eval_agent_group.to("cpu")
        self.eval_critic.to("cpu")
        self.target_agent_group.to("cpu")
        self.target_critic.to("cpu")

        torch.cuda.empty_cache()

        return total_loss / total_batches

    def _learn_multi_gpu(self, sample_size, batch_size: int, times: int = 1):
        """Multi-GPU placeholder.  Raise a clear error."""
        raise NotImplementedError(
            "QPLEX multi-GPU training is not yet implemented. "
            "Please set ``train_device`` to a single device."
        )
