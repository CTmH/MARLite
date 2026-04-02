"""
Base worker module for multi-GPU training.

This module provides the BaseWorker class that serves as the foundation
for all worker implementations. Each worker runs in a separate process
and holds copies of models for parallel training.
"""

import io
import os
import torch
import torch.distributed as dist
from copy import deepcopy
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

        self.assigned_device = f"cuda:{device_id}"
        self.device = "cpu"
        self._setup_distributed()

        # Model copies (to be initialized by subclasses via set_models)
        self.eval_agent_group = None
        self.target_agent_group = None
        self.eval_critic = None
        self.target_critic = None

        # Optimizers (to be initialized by subclasses)
        self.critic_optimizer: Optional[torch.optim.Optimizer] = None
        self.agent_optimizer: Optional[torch.optim.Optimizer] = None
        self.ssl_optimizer: Optional[torch.optim.Optimizer] = None

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
            local_params = {k: v.clone() for k, v in params["eval_agent_group"].items()}
            self.eval_agent_group.load_state_dict(local_params)
        if "target_agent_group" in params and self.target_agent_group is not None:
            local_params = {
                k: v.clone() for k, v in params["target_agent_group"].items()
            }
            self.target_agent_group.load_state_dict(local_params)
        if "eval_critic" in params and self.eval_critic is not None:
            local_params = {k: v.clone() for k, v in params["eval_critic"].items()}
            self.eval_critic.load_state_dict(local_params)
        if "target_critic" in params and self.target_critic is not None:
            local_params = {k: v.clone() for k, v in params["target_critic"].items()}
            self.target_critic.load_state_dict(local_params)

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

    def get_params_for_main(self) -> Dict[str, Any]:
        """
        Get current parameters to send back to main process.

        Returns:
            Dictionary containing cloned parameter data
        """
        return {
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
            del params  # Release reference to allow garbage collection
            ack_queue.put("ACK")

        elif cmd == "BROADCAST":
            params = param_queue.get()
            self.sync_params_from_main(params)
            del params  # Release reference to allow garbage collection

        elif cmd == "SYNC_TO_MAIN":
            params = self.get_params_for_main()
            param_queue.put(params)

        elif cmd == "TRAIN_STEP":
            batch = data_queue.get()
            loss = self.train_step(batch)
            del batch  # Release reference to allow garbage collection
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
            if "ssl_lr" in lr_data:
                for param_group in self.ssl_optimizer.param_groups:
                    param_group["lr"] = lr_data["ssl_lr"]
            if ack_queue:
                ack_queue.put("ACK")

        else:
            print(f"Worker {self.worker_id}: Unknown command: {repr(cmd)}", flush=True)

        return True

    def cleanup(self):
        """Clean up distributed process group."""
        if dist.is_initialized():
            dist.destroy_process_group()
