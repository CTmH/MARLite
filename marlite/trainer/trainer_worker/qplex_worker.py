"""QPLEX worker — multi-GPU training step.

Mirrors :class:`QMIXWorker` but the critic is a :class:`QPLEXMixer`
that consumes the full per-agent Q-values (``(B, N, A)``) together
with the per-agent action indices and returns the decomposed
``q_tot``, ``v_tot``, ``a_tot`` and ``att_reg``.

The training step follows the same Double-DQN procedure as
:class:`QPLEXTrainer._learn_single_gpu`, but is executed inside a
distributed worker process under :class:`OffPolicyWorkerGroup`.
"""

import torch
from typing import Any, Dict

from marlite.algorithm.agents import AgentGroupConfig
from marlite.algorithm.critic import CriticConfig
from marlite.util.optimizer_config import OptimizerConfig
from marlite.trainer.trainer_worker.offpolicy_worker import OffPolicyWorker


class QPLEXWorker(OffPolicyWorker):
    """Worker for QPLEX multi-GPU training.

    Each worker owns copies of the eval and target agent groups and
    critics, plus their optimisers.  The :meth:`train_step` method
    performs one gradient update on a mini-batch and synchronises
    gradients across all workers via all-reduce.
    """

    critic_optimizer: torch.optim.Optimizer
    agent_optimizer: torch.optim.Optimizer

    def __init__(
        self,
        worker_id: int,
        device_id: int,
        rank: int,
        world_size: int,
        init_method: str,
        agent_group_config: AgentGroupConfig,
        critic_config: CriticConfig,
        critic_optimizer_config: OptimizerConfig,
        agent_optimizer_config: OptimizerConfig,
        gamma: float = 0.9,
        max_grad_norm: float = 5.0,
        **kwargs,
    ):
        # Capture kwargs before initialising the parent (parent calls
        # ``_create_worker_kwargs`` which may require additional args).
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.agent_group_config = agent_group_config
        self.critic_config = critic_config
        self.critic_optimizer_config = critic_optimizer_config
        self.agent_optimizer_config = agent_optimizer_config

        super().__init__(worker_id, device_id, rank, world_size, init_method)

        # Build models.
        self.eval_agent_group = agent_group_config.get_agent_group()
        self.target_agent_group = agent_group_config.get_agent_group()
        self.eval_critic = critic_config.get_critic()
        self.target_critic = critic_config.get_critic()

        self.eval_agent_group.train()
        self.target_agent_group.eval()
        self.eval_critic.train()
        self.target_critic.eval()

        # Build optimisers.
        self.critic_optimizer = self.critic_optimizer_config.get_optimizer(
            self.eval_critic.parameters()
        )
        self.agent_optimizer = self.agent_optimizer_config.get_optimizer(
            self.eval_agent_group.parameters()
        )

    def train_step(self, batch: Dict[str, Any]) -> float:
        """Execute one QPLEX training step on a mini-batch.

        Args:
            batch: A collated batch of trajectory segments.  Expected
                keys: ``alive_mask``, ``observations``,
                ``timestep_padding_mask``, ``states``, ``actions``,
                ``rewards``, ``next_states``, ``next_observations``,
                ``next_timestep_padding_mask``, ``next_avail_actions``,
                ``next_alive_mask``, ``terminations``.

        Returns:
            The scalar loss (detached, on CPU) after backpropagation
            and gradient synchronisation.
        """
        # ------------------------------------------------------------------
        # 1. Load batch.
        # ------------------------------------------------------------------
        alive_mask = batch["alive_mask"].to(dtype=torch.bool)
        observations = batch["observations"].to(dtype=torch.float32)
        timestep_padding_mask = batch["timestep_padding_mask"].to(dtype=torch.bool)
        states = batch["states"].to(dtype=torch.float32)
        actions = batch["actions"].to(dtype=torch.int)
        rewards = batch["rewards"].to(dtype=torch.float32)
        next_states = batch["next_states"].to(dtype=torch.float32)
        next_observations = batch["next_observations"].to(dtype=torch.float32)
        next_timestep_padding_mask = batch["next_timestep_padding_mask"].to(
            dtype=torch.bool
        )
        next_avail_actions = batch["next_avail_actions"]
        next_alive_mask = batch["next_alive_mask"].to(dtype=torch.bool)
        terminations = batch["terminations"].to(dtype=torch.bool)

        bs = states.shape[0]
        n_agents = rewards.shape[2]

        next_alive_mask = next_alive_mask.to(self.device)
        alive_mask = alive_mask.to(self.device)

        if isinstance(next_avail_actions, torch.Tensor):
            use_action_mask = True
            next_avail_actions = next_avail_actions[:, -1, :, :]
            next_avail_actions = next_avail_actions.to(
                dtype=torch.bool, device=self.device
            )
        else:
            use_action_mask = False

        r_last = self._aggregate_rewards(rewards[:, -1]).to(self.device)
        termination_last = terminations[:, -1].prod(dim=-1).to(self.device)

        timestep_padding_mask = torch.stack(
            [timestep_padding_mask] * n_agents, dim=1
        ).to(self.device)
        next_timestep_padding_mask = torch.stack(
            [next_timestep_padding_mask] * n_agents, dim=1
        ).to(self.device)

        # ------------------------------------------------------------------
        # 2. Forward: eval networks.
        # ------------------------------------------------------------------
        self.eval_agent_group.train()
        observations = torch.transpose(observations, 1, 2).to(self.device)
        ret = self.eval_agent_group(
            observations, timestep_padding_mask, alive_mask[:, -1, :]
        )
        q_val = ret["q_val"]  # (B, N, A) — full Q-values.
        actions_last = actions[:, -1].to(device=self.device, dtype=torch.int64)
        states = states.to(self.device)

        self.eval_critic.train()
        cret = self.eval_critic(
            q_val, states, actions_last, alive_mask, timestep_padding_mask[:, 0, :]
        )
        q_tot = cret["q_tot"]
        att_reg = cret["att_reg"]

        # ------------------------------------------------------------------
        # 3. Target (Double DQN).
        # ------------------------------------------------------------------
        with torch.no_grad():
            self.eval_agent_group.eval()
            next_observations = torch.transpose(next_observations, 1, 2).to(self.device)
            ret_next_eval = self.eval_agent_group(
                next_observations,
                next_timestep_padding_mask,
                next_alive_mask[:, -1, :],
            )
            q_val_next_eval = ret_next_eval["q_val"]
            if use_action_mask:
                q_val_next_eval = torch.masked_fill(
                    q_val_next_eval, ~next_avail_actions, -torch.inf
                )
            best_actions = q_val_next_eval.argmax(dim=-1)

            self.target_agent_group.eval()
            ret_next_target = self.target_agent_group(
                next_observations,
                next_timestep_padding_mask,
                next_alive_mask[:, -1, :],
            )
            q_val_next = ret_next_target["q_val"]  # (B, N, A)
            next_states = next_states.to(self.device)
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
        # 4. Loss.
        # ------------------------------------------------------------------
        y_tot = r_last + (1 - termination_last) * self.gamma * q_tot_next
        critic_loss = torch.nn.functional.mse_loss(q_tot, y_tot.detach())

        if att_reg.item() != 0:
            total_loss = critic_loss + att_reg
        else:
            total_loss = critic_loss

        # ------------------------------------------------------------------
        # 5. Backprop + synchronise.
        # ------------------------------------------------------------------
        self.agent_optimizer.zero_grad()
        self.eval_critic.zero_grad()
        total_loss.backward()

        self.reduce_gradients()

        torch.nn.utils.clip_grad_norm_(
            self.eval_critic.parameters(), max_norm=self.max_grad_norm
        )
        torch.nn.utils.clip_grad_norm_(
            self.eval_agent_group.parameters(), max_norm=self.max_grad_norm
        )
        self.critic_optimizer.step()
        self.agent_optimizer.step()

        # Per-batch target update (hard / ema)
        self._update_target_after_batch()

        return total_loss.detach().cpu().item()
