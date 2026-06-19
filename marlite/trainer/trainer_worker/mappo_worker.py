"""
MAPPO worker implementation for multi-GPU training.

This module provides the MAPPOWorker class that implements the PPO training
logic for MAPPO algorithm in a multi-GPU setting. Each worker runs in a
separate process and holds copies of the eval agent group and critic models.
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Any, Dict

from marlite.algorithm.agents import AgentGroupConfig
from marlite.algorithm.critic import CriticConfig
from marlite.util.optimizer_config import OptimizerConfig
from marlite.trainer.trainer_worker.onpolicy_worker import OnPolicyWorker


class MAPPOWorker(OnPolicyWorker):
    """
    Worker for MAPPO algorithm multi-GPU training.

    Implements train_step() method that executes one batch of MAPPO training:
    1. Forward pass through eval_agent_group to get action logits
    2. Forward pass through eval_critic to get state values
    3. Compute GAE advantages and returns
    4. Compute PPO clipped surrogate loss for the actor
    5. Compute MSE value loss for the critic
    6. Backward pass with gradient synchronization across workers
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
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 5.0,
        **kwargs,
    ):
        """
        Initialize MAPPO worker.

        Args:
            worker_id: Unique worker identifier.
            device_id: CUDA device ID.
            rank: Global rank in distributed training.
            world_size: Total number of processes.
            init_method: URL for distributed initialization.
            agent_group_config: Configuration for agent group.
            critic_config: Configuration for critic.
            critic_optimizer_config: Configuration for critic optimizer.
            agent_optimizer_config: Configuration for agent group optimizer.
            gamma: Discount factor.
            clip_epsilon: PPO clip range.
            gae_lambda: GAE lambda parameter.
            entropy_coef: Entropy bonus coefficient.
            vf_coef: Value function loss coefficient.
            max_grad_norm: Maximum gradient norm for clipping.
        """
        super().__init__(worker_id, device_id, rank, world_size, init_method)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.eval_agent_group = agent_group_config.get_agent_group()
        self.eval_critic = critic_config.get_critic()

        self.eval_agent_group.train()
        self.eval_critic.train()

        self.critic_optimizer = critic_optimizer_config.get_optimizer(
            self.eval_critic.parameters()
        )
        self.agent_optimizer = agent_optimizer_config.get_optimizer(
            self.eval_agent_group.parameters()
        )

    def _reduce_agent_gradients(self):
        """Reduce agent group gradients across all workers via all_reduce."""
        for param in self.eval_agent_group.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.world_size

    def _reduce_critic_gradients(self):
        """Reduce critic gradients across all workers via all_reduce."""
        for param in self.eval_critic.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.world_size

    def train_step(self, batch: Dict[str, Any]) -> float:
        """
        Execute one MAPPO training step on the given batch.

        Implements the full PPO training logic:
        - Forward pass through agent group to get action logits
        - Forward pass through critic to get state values
        - Compute GAE advantages and returns
        - PPO clipped surrogate loss for actor with entropy bonus
        - MSE value loss for critic
        - Separate backward passes with gradient synchronization

        Args:
            batch: Dictionary containing batch data with keys:
                alive_mask, observations, timestep_padding_mask, states,
                actions, rewards, next_states, next_timestep_padding_mask,
                next_alive_mask, all_log_probs, terminations.

        Returns:
            Combined loss value (actor + vf_coef * critic).
        """
        alive_mask = batch["alive_mask"].to(dtype=torch.bool)
        observations = batch["observations"].to(dtype=torch.float32)
        timestep_padding_mask = batch["timestep_padding_mask"].to(
            dtype=torch.bool, device=self.device
        )
        states = batch["states"].to(dtype=torch.float32)
        actions = batch["actions"].to(dtype=torch.int)
        rewards = batch["rewards"].to(dtype=torch.float32)
        next_states = batch["next_states"].to(dtype=torch.float32)
        next_timestep_padding_mask = batch["next_timestep_padding_mask"].to(
            dtype=torch.bool, device=self.device
        )
        next_alive_mask = batch["next_alive_mask"].to(dtype=torch.bool)
        all_log_probs = batch["all_log_probs"].to(dtype=torch.float32)
        terminations = batch["terminations"].to(dtype=torch.bool)

        bs = states.shape[0]
        n_agents = rewards.shape[2]
        t_steps = rewards.shape[1]

        alive_mask = alive_mask.to(self.device)
        next_alive_mask = next_alive_mask.to(self.device)
        states_dev = states.to(self.device)
        next_states_dev = next_states.to(self.device)

        rewards_sum = rewards.sum(dim=2).to(self.device)
        terminations_any = terminations.any(dim=2).to(
            dtype=torch.float32, device=self.device
        )

        timestep_padding_mask_expanded = torch.stack(
            [timestep_padding_mask] * n_agents, dim=1
        ).to(self.device)

        self.eval_critic.train()
        v = self.eval_critic(states_dev, alive_mask, timestep_padding_mask)["v"]
        v_last = v[:, 0]  # (B,)

        with torch.no_grad():
            v_next = self.eval_critic(
                next_states_dev[:, -1:, ...],
                next_alive_mask[:, -1:, ...],
                next_timestep_padding_mask[:, -1:],
            )["v"][:, 0]  # (B,)

        r_last = self._aggregate_rewards(rewards[:, -1]).to(self.device)
        termination_last = terminations[:, -1].prod(dim=-1).to(
            dtype=torch.float32, device=self.device
        )
        delta = r_last + self.gamma * v_next * (1.0 - termination_last) - v_last
        advantages_last = delta  # (B,)
        returns = delta + v_last

        observations_transposed = torch.transpose(observations, 1, 2).to(self.device)
        self.eval_agent_group.train()
        ret_agent = self.eval_agent_group(
            observations_transposed,
            timestep_padding_mask_expanded,
            alive_mask[:, -1, :],
        )
        action_logits = ret_agent["action_logits"]

        actions_last = actions[:, -1].to(dtype=torch.int64, device=self.device)
        log_probs_old = all_log_probs[:, -1, :].to(self.device)

        dist = Categorical(logits=action_logits)
        new_log_probs = dist.log_prob(actions_last)
        entropy = dist.entropy()

        alive_last_flag = alive_mask[:, -1, :].to(
            dtype=torch.float32, device=self.device
        )
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
            / max(alive_last_count, torch.tensor(1.0, device=self.device))
        )
        entropy_loss = (
            -(entropy * alive_last_flag).sum()
            / max(alive_last_count, torch.tensor(1.0, device=self.device))
        )
        actor_loss = actor_loss + self.entropy_coef * entropy_loss

        critic_loss = F.mse_loss(v_last, returns.detach())

        self.agent_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self._reduce_agent_gradients()
        torch.nn.utils.clip_grad_norm_(
            self.eval_agent_group.parameters(), max_norm=self.max_grad_norm
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self._reduce_critic_gradients()
        torch.nn.utils.clip_grad_norm_(
            self.eval_critic.parameters(), max_norm=self.max_grad_norm
        )

        self.agent_optimizer.step()
        self.critic_optimizer.step()

        combined_loss = actor_loss.detach().cpu().item() + self.vf_coef * critic_loss.detach().cpu().item()
        return combined_loss
