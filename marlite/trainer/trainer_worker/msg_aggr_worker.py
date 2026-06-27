"""
Message aggregation worker implementation for multi-GPU training.

This module provides the MsgAggrWorker and ProbMsgAggrWorker classes that implement
the training logic for message aggregation algorithms in a multi-GPU setting.
"""

import torch
import torch.distributed as dist
from typing import Any, Dict

from marlite.algorithm.agents import AgentGroupConfig
from marlite.algorithm.critic import CriticConfig
from marlite.util.optimizer_config import OptimizerConfig
from marlite.trainer.trainer_worker.offpolicy_worker import OffPolicyWorker


class MsgAggrWorker(OffPolicyWorker):
    """
    Worker for MsgAggrQMIX algorithm multi-GPU training.

    Implements train_step() method with message aggregation loss.
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
        warmup_epochs: int = 0,
        msg_aggr_weight: float = 1.0,
        **kwargs,
    ):
        """
        Initialize MsgAggr worker.

        Args:
            worker_id: Unique worker identifier
            device_id: CUDA device ID
            rank: Global rank in distributed training
            world_size: Total number of processes
            init_method: URL for distributed initialization
            agent_group_config: Configuration for agent group
            critic_config: Configuration for critic
            critic_optimizer_config: Configuration for critic optimizer
            agent_optimizer_config: Configuration for agent group optimizer
            gamma: Discount factor
            max_grad_norm: Maximum gradient norm for clipping
            warmup_epochs: Number of warmup epochs before message aggregation loss is used
            msg_aggr_weight: Weight for message aggregation loss
        """
        super().__init__(worker_id, device_id, rank, world_size, init_method)
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.warmup_epochs = warmup_epochs
        self.msg_aggr_weight = msg_aggr_weight

        self.eval_agent_group = agent_group_config.get_agent_group()
        self.target_agent_group = agent_group_config.get_agent_group()
        self.eval_critic = critic_config.get_critic()
        self.target_critic = critic_config.get_critic()

        self.eval_agent_group.train()
        self.target_agent_group.eval()
        self.eval_critic.train()
        self.target_critic.eval()

        self.critic_optimizer = critic_optimizer_config.get_optimizer(
            self.eval_critic.parameters()
        )
        self.agent_optimizer = agent_optimizer_config.get_optimizer(
            self.eval_agent_group.parameters()
        )

    def train_step(self, batch: Dict[str, Any]) -> float:
        """
        Execute one training step on the given batch.

        Implements MsgAggrQMIX training logic with optional message aggregation loss.

        Args:
            batch: Dictionary containing batch data

        Returns:
            loss: Computed critic loss value
        """
        # Extract batch data
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
        terminations = batch["terminations"].to(dtype=torch.bool)
        truncations = batch["truncations"].to(dtype=torch.bool)

        bs = states.shape[0]
        n_agents = rewards.shape[2]

        # Create alive_mask_next from terminations and truncations
        done_flags = terminations[:, -1]
        truncations = truncations[:, -1]
        next_alive_mask = ~(done_flags | truncations)
        next_alive_mask = next_alive_mask.unsqueeze(dim=1)
        next_alive_mask = torch.cat([alive_mask[:, 1:, :], next_alive_mask], dim=1)
        next_alive_mask = next_alive_mask.to(self.device)
        alive_mask = alive_mask.to(self.device)

        # Handle action mask
        if isinstance(next_avail_actions, torch.Tensor):
            use_action_mask = True
            next_avail_actions = next_avail_actions[:, -1, :, :]
            next_avail_actions = next_avail_actions.to(
                dtype=torch.bool, device=self.device
            )
        else:
            use_action_mask = False

        # Process rewards and terminations
        r_last = self._aggregate_rewards(rewards[:, -1]).to(self.device)
        termination_last = done_flags.prod(dim=-1).to(self.device)

        # Process padding masks
        timestep_padding_mask = torch.stack(
            [timestep_padding_mask] * n_agents, dim=1
        ).to(self.device)
        next_timestep_padding_mask = torch.stack(
            [next_timestep_padding_mask] * n_agents, dim=1
        ).to(self.device)

        # Compute Q-tot for current state
        self.eval_agent_group.train()
        observations = torch.transpose(observations, 1, 2).to(self.device)
        ret = self.eval_agent_group(
            observations, timestep_padding_mask, alive_mask[:, -1, :]
        )
        q_val = ret["q_val"]
        aggregated_msg = ret["aggregated_msg"]
        actions = actions[:, -1].to(device=self.device, dtype=torch.int64)
        q_val = torch.gather(q_val, dim=-1, index=actions.unsqueeze(-1))
        q_val = q_val.squeeze(-1)
        states = states.to(self.device)
        self.eval_critic.train()
        ret = self.eval_critic(
            q_val, states, alive_mask, timestep_padding_mask[:, 0, :]
        )
        q_tot = ret["q_tot"]

        with torch.no_grad():
            self.target_critic.eval()
            ret = self.target_critic(
                q_val, states, alive_mask, timestep_padding_mask[:, 0, :]
            )
            state_features = ret["state_features"]

        # Use target model for stability (Double Q-learning)
        with torch.no_grad():
            # Double Q: eval agent group selects best actions
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

            # Double Q: target agent group evaluates chosen actions
            self.target_agent_group.eval()
            ret_next_target = self.target_agent_group(
                next_observations,
                next_timestep_padding_mask,
                next_alive_mask[:, -1, :],
            )
            q_val_next = ret_next_target["q_val"].gather(
                dim=-1, index=best_actions.unsqueeze(-1)
            ).squeeze(-1)
            next_states = next_states.to(self.device)
            self.target_critic.eval()
            ret_next = self.target_critic(
                q_val_next,
                next_states,
                next_alive_mask,
                next_timestep_padding_mask[:, 0, :],
            )
            q_tot_next = ret_next["q_tot"]

        # Compute TD target
        y_tot = r_last + (1 - termination_last) * self.gamma * q_tot_next

        # Compute TD error
        td_error = torch.nn.functional.mse_loss(q_tot, y_tot.detach())

        # Only compute message aggregation losses after warmup period
        current_epoch = batch.get("epoch", 0)
        if current_epoch >= self.warmup_epochs:
            msg_aggr_loss = torch.nn.functional.smooth_l1_loss(
                aggregated_msg, state_features.detach()
            )
            critic_loss = td_error + self.msg_aggr_weight * msg_aggr_loss
        else:
            critic_loss = td_error

        # Backward pass
        self.agent_optimizer.zero_grad()
        self.eval_critic.zero_grad()
        critic_loss.backward()

        # Synchronize gradients across all workers
        self.reduce_gradients()

        # Clip gradients and optimize
        torch.nn.utils.clip_grad_norm_(self.eval_critic.parameters(), max_norm=self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.eval_agent_group.parameters(), max_norm=self.max_grad_norm)
        self.critic_optimizer.step()
        self.agent_optimizer.step()

        # Per-batch target update (hard / ema / polyak)
        self._update_target_after_batch()

        return critic_loss.detach().cpu().item()


class ProbMsgAggrWorker(MsgAggrWorker):
    """
    Worker for ProbMsgAggrQMIX algorithm multi-GPU training.

    Extends MsgAggrWorker with probabilistic message aggregation using KL divergence.
    """

    def __init__(
        self,
        worker_id: int,
        device_id: int,
        rank: int,
        world_size: int,
        init_method: str,
        agent_group_config=None,
        critic_config=None,
        critic_optimizer_config=None,
        agent_optimizer_config=None,
        gamma: float = 0.9,
        max_grad_norm: float = 5.0,
        warmup_epochs: int = 0,
        msg_aggr_weight: float = 1.0,
        **kwargs,
    ):
        """
        Initialize ProbMsgAggr worker.

        Args:
            worker_id: Unique worker identifier
            device_id: CUDA device ID
            rank: Global rank in distributed training
            world_size: Total number of processes
            init_method: URL for distributed initialization
            agent_group_config: Configuration for agent group
            critic_config: Configuration for critic
            critic_optimizer_config: Configuration for critic optimizer
            agent_optimizer_config: Configuration for agent group optimizer
            gamma: Discount factor
            max_grad_norm: Maximum gradient norm for clipping
            warmup_epochs: Number of warmup epochs before message aggregation loss is used
            msg_aggr_weight: Weight for message aggregation loss
        """
        super().__init__(
            worker_id,
            device_id,
            rank,
            world_size,
            init_method,
            agent_group_config,
            critic_config,
            critic_optimizer_config,
            agent_optimizer_config,
            gamma,
            max_grad_norm,
            warmup_epochs,
            msg_aggr_weight,
            **kwargs,
        )
        from torch.distributions import Normal, kl_divergence

        self.Normal = Normal
        self.kl_divergence = kl_divergence

    def train_step(self, batch: Dict[str, Any]) -> float:
        """
        Execute one training step with probabilistic message aggregation.

        Args:
            batch: Dictionary containing batch data

        Returns:
            loss: Computed critic loss value
        """
        # Extract batch data
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
        truncations = batch["truncations"].to(dtype=torch.bool)

        bs = states.shape[0]
        n_agents = rewards.shape[2]

        # Prepare masks and move to device
        next_alive_mask = next_alive_mask.to(self.device)
        alive_mask = alive_mask.to(self.device)

        # Handle action mask
        if isinstance(next_avail_actions, torch.Tensor):
            use_action_mask = True
            next_avail_actions = next_avail_actions[:, -1, :, :]
            next_avail_actions = next_avail_actions.to(
                dtype=torch.bool, device=self.device
            )
        else:
            use_action_mask = False

        # Process rewards and terminations
        r_last = self._aggregate_rewards(rewards[:, -1]).to(self.device)
        termination_last = terminations[:, -1].prod(dim=-1).to(self.device)

        # Process padding masks
        timestep_padding_mask = torch.stack(
            [timestep_padding_mask] * n_agents, dim=1
        ).to(self.device)
        next_timestep_padding_mask = torch.stack(
            [next_timestep_padding_mask] * n_agents, dim=1
        ).to(self.device)

        # Compute Q-tot for current state
        self.eval_agent_group.train()
        observations = torch.transpose(observations, 1, 2).to(self.device)
        ret = self.eval_agent_group(
            observations, timestep_padding_mask, alive_mask[:, -1, :]
        )
        q_val = ret["q_val"]
        ag_mu = ret["mu"]
        ag_std = ret["std"]
        actions = actions[:, -1].to(device=self.device, dtype=torch.int64)
        q_val = torch.gather(q_val, dim=-1, index=actions.unsqueeze(-1))
        q_val = q_val.squeeze(-1)
        states = states.to(self.device)
        self.eval_critic.train()
        ret = self.eval_critic(
            q_val, states, alive_mask, timestep_padding_mask[:, 0, :]
        )
        q_tot = ret["q_tot"]

        # Use target model for stability (Double Q-learning)
        with torch.no_grad():
            # Get critic's distribution for current state (for KL)
            self.target_critic.eval()
            ret_critic = self.target_critic(
                q_val, states, alive_mask, timestep_padding_mask[:, 0, :]
            )
            critic_mu = ret_critic["mu"]
            critic_std = ret_critic["std"]

            # Double Q: eval agent group selects best actions
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

            # Double Q: target agent group evaluates chosen actions
            self.target_agent_group.eval()
            ret_next_target = self.target_agent_group(
                next_observations,
                next_timestep_padding_mask,
                next_alive_mask[:, -1, :],
            )
            q_val_next = ret_next_target["q_val"].gather(
                dim=-1, index=best_actions.unsqueeze(-1)
            ).squeeze(-1)
            next_states = next_states.to(self.device)
            self.target_critic.eval()
            ret_next = self.target_critic(
                q_val_next,
                next_states,
                next_alive_mask,
                next_timestep_padding_mask[:, 0, :],
            )
            q_tot_next = ret_next["q_tot"]

        # Compute TD target
        y_tot = r_last + (1 - termination_last) * self.gamma * q_tot_next

        # Compute TD error
        td_error = torch.nn.functional.mse_loss(q_tot, y_tot.detach())

        # Message aggregation loss using KL divergence
        ag_distribution = self.Normal(ag_mu, ag_std)
        critic_distribution = self.Normal(critic_mu.detach(), critic_std.detach())
        msg_aggr_loss = self.kl_divergence(ag_distribution, critic_distribution).mean()

        # Use combined loss
        current_epoch = batch.get("epoch", 0)
        if current_epoch >= self.warmup_epochs:
            critic_loss = td_error + self.msg_aggr_weight * msg_aggr_loss
        else:
            critic_loss = td_error

        # Backward pass
        self.agent_optimizer.zero_grad()
        self.eval_critic.zero_grad()
        critic_loss.backward()

        # Synchronize gradients across all workers
        self.reduce_gradients()

        # Clip gradients and optimize
        torch.nn.utils.clip_grad_norm_(self.eval_critic.parameters(), max_norm=self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.eval_agent_group.parameters(), max_norm=self.max_grad_norm)
        self.critic_optimizer.step()
        self.agent_optimizer.step()

        # Per-batch target update (hard / ema / polyak)
        self._update_target_after_batch()

        return critic_loss.detach().cpu().item()
