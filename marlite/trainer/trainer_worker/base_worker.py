"""
Base worker module for multi-GPU training.

This module provides the BaseWorker class that serves as the foundation
for all worker implementations. Each worker runs in a separate process
and holds copies of models for parallel training.
"""

import os
import torch
import torch.distributed as dist
from typing import Any, Dict, Optional


class BaseWorker:
    """
    Base worker class for multi-GPU training.

    Each worker runs in a separate process and maintains its own copies of:
    - eval_agent_group
    - target_agent_group
    - eval_critic
    - target_critic

    Workers communicate with the main process via multiprocessing queues
    for parameter synchronization and training coordination.
    """

    def __init__(
        self,
        worker_id: int,
        device_id: int,
        rank: int,
        world_size: int,
        init_method: str,
    ):
        """
        Initialize the base worker.

        Args:
            worker_id: Unique identifier for this worker (0 to num_workers-1)
            device_id: CUDA device ID to use (local rank)
            rank: Global rank in distributed training
            world_size: Total number of processes
            init_method: URL specifying how to initialize distributed training
        """
        self.worker_id = worker_id
        self.device_id = device_id
        self.rank = rank
        self.world_size = world_size
        self.init_method = init_method

        self.device = f"cuda:{device_id}"
        self._setup_distributed()

        # Model copies (to be initialized by subclasses via set_models)
        self.eval_agent_group = None
        self.target_agent_group = None
        self.eval_critic = None
        self.target_critic = None

        # Optimizers (to be initialized by subclasses via set_optimizers)
        self.critic_optimizer = None
        self.agent_group_optimizer = None

        # Parameter version for tracking updates
        self.params_version = 0

    def _setup_distributed(self):
        """Initialize distributed communication for this worker."""
        torch.cuda.set_device(self.device_id)

        # Initialize distributed process group for this worker
        dist.init_process_group(
            backend="nccl",
            init_method=self.init_method,
            rank=self.rank,
            world_size=self.world_size,
        )

    def set_models(
        self,
        eval_agent_group: Any,
        target_agent_group: Any,
        eval_critic: torch.nn.Module,
        target_critic: torch.nn.Module,
    ):
        """
        Set model instances for this worker.

        Args:
            eval_agent_group: Evaluation agent group instance
            target_agent_group: Target agent group instance
            eval_critic: Evaluation critic module
            target_critic: Target critic module
        """
        self.eval_agent_group = eval_agent_group.to(self.device)
        self.target_agent_group = target_agent_group.to(self.device)
        self.eval_critic = eval_critic.to(self.device)
        self.target_critic = target_critic.to(self.device)

        # Set models to appropriate mode
        self.eval_agent_group.train()
        self.target_agent_group.eval()
        self.eval_critic.train()
        self.target_critic.eval()

    def set_optimizers(
        self,
        critic_optimizer: torch.optim.Optimizer,
        agent_group_optimizer: torch.optim.Optimizer,
    ):
        """
        Set optimizer instances for this worker.

        Args:
            critic_optimizer: Optimizer for critic parameters
            agent_group_optimizer: Optimizer for agent group parameters
        """
        self.critic_optimizer = critic_optimizer
        self.agent_group_optimizer = agent_group_optimizer

    def sync_params_from_shared_memory(self, shared_memory: Dict[str, Any]):
        """
        Synchronize parameters from shared memory.

        Args:
            shared_memory: Dictionary containing parameter data
        """
        if "eval_agent_group" in shared_memory:
            self.eval_agent_group.set_agent_group_params(
                shared_memory["eval_agent_group"]
            )
        if "target_agent_group" in shared_memory:
            self.target_agent_group.set_agent_group_params(
                shared_memory["target_agent_group"]
            )
        if "eval_critic" in shared_memory:
            self.eval_critic.load_state_dict(shared_memory["eval_critic"])
        if "target_critic" in shared_memory:
            self.target_critic.load_state_dict(shared_memory["target_critic"])
        if "version" in shared_memory:
            self.params_version = shared_memory["version"]

    def write_params_to_shared_memory(self, shared_memory: Dict[str, Any]):
        """
        Write current parameters to shared memory for main process to read.

        Args:
            shared_memory: Dictionary to store parameter data
        """
        shared_memory["eval_agent_group"] = (
            self.eval_agent_group.get_agent_group_params()
        )
        shared_memory["target_agent_group"] = (
            self.target_agent_group.get_agent_group_params()
        )
        shared_memory["eval_critic"] = self.eval_critic.state_dict()
        shared_memory["target_critic"] = self.target_critic.state_dict()
        shared_memory["version"] = self.params_version

    def reduce_gradients(self):
        """
        Reduce (average) gradients across all workers using all_reduce.

        This implements DDP-style gradient synchronization:
        - All workers compute local gradients
        - all_reduce sums gradients across all workers
        - Divide by world_size to get average
        """
        for param in self.eval_critic.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.world_size

        for param in self.eval_agent_group.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.world_size

    def train_step(self, batch: Dict[str, Any]) -> float:
        """
        Execute one training step on the given batch.

        This method should be overridden by subclasses to implement
        specific training logic (QMIX, GraphQMIX, etc.).

        Args:
            batch: Dictionary containing batch data

        Returns:
            loss: Computed loss value for this worker
        """
        raise NotImplementedError("Subclasses must implement train_step()")

    def handle_command(
        self,
        cmd: str,
        param_queue,
        data_queue,
        loss_queue,
    ) -> bool:
        """
        Handle a command from the main process.

        Args:
            cmd: Command string
            param_queue: Queue for parameter exchange
            data_queue: Queue for training data
            loss_queue: Queue for returning loss values

        Returns:
            True if should continue, False if should stop
        """
        if cmd == "STOP":
            self.cleanup()
            return False

        elif cmd == "SYNC_FROM_MAIN":
            # Main process is sending initial parameters
            shared_memory = param_queue.get()
            self.sync_params_from_shared_memory(shared_memory)
            param_queue.put("ACK")

        elif cmd == "BROADCAST":
            # Receiving broadcasted parameters from main
            shared_memory = param_queue.get()
            self.sync_params_from_shared_memory(shared_memory)

        elif cmd == "SYNC_TO_MAIN":
            # Main process wants to read our parameters
            shared_memory = {}
            self.write_params_to_shared_memory(shared_memory)
            param_queue.put(shared_memory)

        elif cmd == "TRAIN_STEP":
            # Execute training step
            batch = data_queue.get()
            loss = self.train_step(batch)
            loss_queue.put(loss)

        return True

    def cleanup(self):
        """Clean up distributed process group."""
        if dist.is_initialized():
            dist.destroy_process_group()
