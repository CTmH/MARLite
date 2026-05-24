"""
Graph worker implementation for multi-GPU training.

This module provides the GraphWorker class that implements the training logic
for GraphQMIX algorithm in a multi-GPU setting.
"""

import io
import torch
import torch.distributed as dist
from typing import Any, Dict

from marlite.algorithm.agents import AgentGroupConfig
from marlite.algorithm.critic import CriticConfig
from marlite.util.optimizer_config import OptimizerConfig
from marlite.trainer.trainer_worker.offpolicy_worker import OffPolicyWorker


class GraphWorker(OffPolicyWorker):
    """
    Worker for GraphQMIX algorithm multi-GPU training.

    Implements train_step() method that executes one batch of GraphQMIX training.
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
        """
        Initialize Graph worker.

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
        """
        super().__init__(worker_id, device_id, rank, world_size, init_method)
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm

        # Initialize RL models
        self.eval_agent_group = agent_group_config.get_agent_group()
        self.target_agent_group = agent_group_config.get_agent_group()
        self.eval_critic = critic_config.get_critic()
        self.target_critic = critic_config.get_critic()

        self.eval_agent_group.train()
        self.target_agent_group.eval()
        self.eval_critic.train()
        self.target_critic.eval()

        # Initialize RL optimizers
        self.critic_optimizer = critic_optimizer_config.get_optimizer(
            self.eval_critic.parameters()
        )
        self.agent_optimizer = agent_optimizer_config.get_optimizer(
            self.eval_agent_group.parameters()
        )

    def move_to_device(self, device: str):
        """
        Move all models to the specified device.

        Args:
            device: Target device string (e.g., 'cuda:0' or 'cpu')
        """
        if self.eval_agent_group is not None:
            self.eval_agent_group.to(device)
        if self.target_agent_group is not None:
            self.target_agent_group.to(device)
        if self.eval_critic is not None:
            self.eval_critic.to(device)
        if self.target_critic is not None:
            self.target_critic.to(device)
        self.device = device

    def reduce_gradients(self):
        """
        Reduce (average) gradients across all workers for critic and agent.

        Implements DDP-style gradient synchronization:
        - All workers compute local gradients
        - all_reduce sums gradients across all workers
        - Divide by world_size to get average
        """
        # Critic gradients
        for param in self.eval_critic.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.world_size

        # Agent group gradients
        for param in self.eval_agent_group.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.world_size

    def get_params_for_main(self) -> Dict[str, Any]:
        """
        Get current parameters to send back to main process.

        Returns:
            Dictionary containing cloned parameter data
        """
        params = {
            "eval_agent_group": {
                k: v.clone().cpu()
                for k, v in self.eval_agent_group.state_dict().items()
            },
            "target_agent_group": {
                k: v.clone().cpu()
                for k, v in self.target_agent_group.state_dict().items()
            },
            "eval_critic": {
                k: v.clone().cpu() for k, v in self.eval_critic.state_dict().items()
            },
            "target_critic": {
                k: v.clone().cpu() for k, v in self.target_critic.state_dict().items()
            },
        }
        return params

    def sync_params_from_main(self, params):
        """
        Synchronize parameters received from main process.

        This method accepts either a dictionary of parameters or serialized bytes.
        When receiving bytes, it deserializes them and loads into local models.

        Args:
            params: Dictionary containing parameter data, or serialized bytes
        """
        # Handle serialized bytes
        if isinstance(params, bytes):
            buffer = io.BytesIO(params)
            params = torch.load(buffer, weights_only=True)

        if "eval_agent_group" in params and self.eval_agent_group is not None:
            self.eval_agent_group.load_state_dict(
                {k: v.clone() for k, v in params["eval_agent_group"].items()}
            )
        if "target_agent_group" in params and self.target_agent_group is not None:
            self.target_agent_group.load_state_dict(
                {k: v.clone() for k, v in params["target_agent_group"].items()}
            )
        if "eval_critic" in params and self.eval_critic is not None:
            self.eval_critic.load_state_dict(
                {k: v.clone() for k, v in params["eval_critic"].items()}
            )
        if "target_critic" in params and self.target_critic is not None:
            self.target_critic.load_state_dict(
                {k: v.clone() for k, v in params["target_critic"].items()}
            )

    def train_step(self, batch: Dict[str, Any]) -> float:
        """
        Execute one training step on the given batch.

        Args:
            batch: Dictionary containing:
                - alive_mask: Agent alive masks
                - observations: Observation sequences (B, T, N, O)
                - timestep_padding_mask: Padding masks
                - states: State sequences (B, T, S)
                - edge_indices: Graph edge indices (list of arrays)
                - actions: Action sequences
                - rewards: Reward sequences
                - next_states: Next state sequences
                - next_observations: Next observation sequences
                - next_edge_indices: Edge indices for next states
                - next_timestep_padding_mask: Padding masks for next states
                - next_avail_actions: Available actions for next states
                - next_alive_mask: Alive masks for next states
                - terminations: Termination flags

        Returns:
            Critic loss (TD error)
        """
        # Extract batch data
        alive_mask = batch["alive_mask"].to(dtype=torch.bool)
        observations = batch["observations"].to(dtype=torch.float32)  # (B, T, N, O)
        timestep_padding_mask = batch["timestep_padding_mask"].to(dtype=torch.bool)
        states = batch["states"].to(dtype=torch.float32)  # (B, T, S)
        edge_indices = batch["edge_indices"]
        actions = batch["actions"].to(dtype=torch.int)
        rewards = batch["rewards"].to(dtype=torch.float32)
        next_states = batch["next_states"].to(dtype=torch.float32)
        next_observations = batch["next_observations"].to(dtype=torch.float32)
        next_edge_indices = batch["next_edge_indices"]
        next_timestep_padding_mask = batch["next_timestep_padding_mask"].to(
            dtype=torch.bool
        )
        next_avail_actions = batch["next_avail_actions"]
        next_alive_mask = batch["next_alive_mask"].to(dtype=torch.bool)
        terminations = batch["terminations"].to(dtype=torch.bool)

        bs = states.shape[0]  # Batch size
        n_agents = rewards.shape[2]  # Number of agents

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
        rewards = rewards[:, -1]  # (B, T, N) -> (B, N)
        rewards = rewards.sum(dim=1).to(self.device)  # (B, N) -> (B)
        terminations = terminations[:, -1]  # (B, T, N) -> (B, N)
        terminations = terminations.prod(dim=1).to(self.device)  # (B, N) -> (B)

        # Process padding masks - expand to (B, N, T)
        timestep_padding_mask = torch.stack(
            [timestep_padding_mask] * n_agents, dim=1
        ).to(self.device)
        next_timestep_padding_mask = torch.stack(
            [next_timestep_padding_mask] * n_agents, dim=1
        ).to(self.device)

        # Extract last edge indices for current and next states
        last_edge_indices = [edge_indices[i][-1] for i in range(bs)]
        last_next_edge_indices = [next_edge_indices[i][-1] for i in range(bs)]

        # === RL Forward Pass ===
        self.eval_agent_group.reset().train()
        # Transpose: (B, T, N, O) -> (B, N, T, O)
        observations_transposed = torch.transpose(observations, 1, 2).to(self.device)
        states = states.to(self.device)

        ret = self.eval_agent_group(
            observations_transposed,
            states,
            timestep_padding_mask,
            alive_mask[:, -1, :],
            last_edge_indices,
        )
        q_val = ret["q_val"]
        # Get actions at last timestep: (B, T, N, A) -> (B, N, A)
        actions_last = actions[:, -1].to(device=self.device, dtype=torch.int64)
        q_val = torch.gather(q_val, dim=-1, index=actions_last.unsqueeze(-1)).squeeze(
            -1
        )
        # q_val: (B, N)

        self.eval_critic.train()
        ret_critic = self.eval_critic(
            q_val, states, alive_mask, timestep_padding_mask[:, 0, :]
        )
        q_tot = ret_critic["q_tot"]
        # q_tot: (B,)

        # === Compute TD Targets ===
        with torch.no_grad():
            self.target_agent_group.reset().eval()
            next_observations_transposed = torch.transpose(next_observations, 1, 2).to(
                self.device
            )
            next_states = next_states.to(self.device)
            ret_next = self.target_agent_group(
                next_observations_transposed,
                next_states,
                next_timestep_padding_mask,
                next_alive_mask[:, -1, :],
                last_next_edge_indices,
            )
            q_val_next = ret_next["q_val"]
            if use_action_mask:
                q_val_next = torch.masked_fill(
                    q_val_next, ~next_avail_actions, -torch.inf
                )
            q_val_next = q_val_next.max(dim=-1).values
            # q_val_next: (B, N)

            self.target_critic.eval()
            ret_next_critic = self.target_critic(
                q_val_next,
                next_states,
                next_alive_mask,
                next_timestep_padding_mask[:, 0, :],
            )
            q_tot_next = ret_next_critic["q_tot"]
            # q_tot_next: (B,)

        # Compute TD target: y_tot = r + gamma * (1 - terminations) * q_tot_next
        y_tot = rewards + (1 - terminations) * self.gamma * q_tot_next

        # Compute critic loss (TD error)
        critic_loss = torch.nn.functional.mse_loss(q_tot, y_tot.detach())

        # === Backward Pass ===
        self.critic_optimizer.zero_grad()
        self.agent_optimizer.zero_grad()

        critic_loss.backward()

        # Synchronize gradients across all workers
        self.reduce_gradients()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.eval_critic.parameters(), max_norm=self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.eval_agent_group.parameters(), max_norm=self.max_grad_norm)

        # Optimizer steps
        self.critic_optimizer.step()
        self.agent_optimizer.step()

        return critic_loss.detach().cpu().item()

    def handle_command(
        self,
        cmd: str,
        param_queue,
        data_queue,
        loss_queue,
        ack_queue=None,
    ) -> bool:
        """
        Handle a command from the main process.

        Args:
            cmd: Command string
            param_queue: Queue for parameter exchange
            data_queue: Queue for receiving training data
            loss_queue: Queue for returning loss values
            ack_queue: Queue for sending ACK signals back to main process

        Returns:
            True if should continue, False if should stop
        """
        if cmd == "STOP":
            self.cleanup()
            return False

        elif cmd == "SYNC_FROM_MAIN":
            params = param_queue.get()
            self.sync_params_from_main(params)
            del params
            ack_queue.put("ACK")

        elif cmd == "BROADCAST":
            params = param_queue.get()
            self.sync_params_from_main(params)
            del params

        elif cmd == "SYNC_TO_MAIN":
            params = self.get_params_for_main()
            param_queue.put(params)

        elif cmd == "TRAIN_STEP":
            batch = data_queue.get()
            loss = self.train_step(batch)
            del batch
            loss_queue.put(loss)

        elif cmd == "MOVE_TO_GPU":
            self.move_to_device(self.assigned_device)
            if ack_queue:
                ack_queue.put("ACK")

        elif cmd == "MOVE_TO_CPU":
            self.move_to_device("cpu")
            torch.cuda.empty_cache()
            if ack_queue:
                ack_queue.put("ACK")

        elif cmd == "SYNC_LR":
            lr_data = param_queue.get()
            if "critic_lr" in lr_data:
                for param_group in self.critic_optimizer.param_groups:
                    param_group["lr"] = lr_data["critic_lr"]
            if "agent_lr" in lr_data:
                for param_group in self.agent_optimizer.param_groups:
                    param_group["lr"] = lr_data["agent_lr"]
            if ack_queue:
                ack_queue.put("ACK")

        else:
            print(f"Worker {self.worker_id}: Unknown command: {repr(cmd)}", flush=True)

        return True

    def cleanup(self):
        """Clean up distributed process group."""
        if dist.is_initialized():
            dist.destroy_process_group()
