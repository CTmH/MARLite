"""
Base worker group module for multi-GPU training.

This module provides the BaseWorkerGroup class that manages multiple worker processes
for parallel training across multiple GPUs.
"""

import os
import torch
import torch.multiprocessing as mp
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod


def worker_loop(
    worker_id,
    device_id,
    rank,
    world_size,
    init_method,
    worker_class,
    worker_kwargs,
    param_queue,
    data_queue,
    loss_queue,
    cmd_queue,
    ready_event,
):
    """
    Main loop function that runs in each worker process.

    Workers wait for commands from the main process and execute them:
    - STOP: Exit the worker loop
    - SYNC_FROM_MAIN: Receive initial parameters from main process
    - BROADCAST: Receive broadcasted parameters from main process
    - SYNC_TO_MAIN: Write current parameters to shared memory for main process
    - TRAIN_STEP: Execute one training step on received batch data

    Args:
        worker_id: Unique worker identifier
        device_id: CUDA device ID for this worker
        rank: Global rank in distributed training
        world_size: Total number of processes
        init_method: URL for distributed initialization
        worker_class: Class of the worker to instantiate
        worker_kwargs: Keyword arguments for worker initialization
        param_queue: Queue for parameter synchronization
        data_queue: Queue for receiving training data
        loss_queue: Queue for sending loss values back to main process
        cmd_queue: Queue for receiving commands
        ready_event: Event to signal worker is ready
    """
    # Create worker instance
    worker = worker_class(**worker_kwargs)

    # Signal that worker is ready
    ready_event.set()

    # Main worker loop
    while True:
        cmd = cmd_queue.get()

        # Handle command via worker's handle_command method
        should_continue = worker.handle_command(
            cmd, param_queue, data_queue, loss_queue
        )

        if not should_continue:
            break


class BaseWorkerGroup(ABC):
    """
    Base worker group for managing multiple GPU workers.

    This class handles:
    - Starting and managing worker processes
    - Parameter synchronization via shared memory
    - Distributing training batches to workers
    - Collecting loss values from workers

    Subclasses should implement:
    - _create_worker(): Create worker instance with proper model setup
    - _get_worker_class(): Return the worker class to use
    """

    def __init__(
        self,
        device_ids: List[int],
        world_size: int,
        init_method: str,
    ):
        """
        Initialize the worker group.

        Args:
            device_ids: List of CUDA device IDs to use
            world_size: Total number of processes (should match len(device_ids))
            init_method: URL for distributed initialization
        """
        self.device_ids = device_ids
        self.world_size = world_size
        self.init_method = init_method

        # Multiprocessing context
        self.mp_ctx = mp.get_context("spawn")

        # Worker processes and queues
        self.workers = []
        self.param_queue = None
        self.data_queue = None
        self.loss_queue = None
        self.cmd_queue = None
        self.ready_events = []

    def _create_worker_kwargs(self) -> Dict[str, Any]:
        """
        Create keyword arguments for worker initialization.

        Returns:
            Dictionary of keyword arguments for the worker class
        """
        return {}

    @abstractmethod
    def _get_worker_class(self):
        """
        Get the worker class to use for this worker group.

        Returns:
            Worker class
        """
        pass

    def start_workers(self):
        """
        Start all worker processes.

        Each worker:
        1. Initializes distributed communication
        2. Creates model copies on its assigned GPU
        3. Waits for commands from main process
        """
        self.param_queue = self.mp_ctx.Queue()
        self.data_queue = self.mp_ctx.Queue()
        self.loss_queue = self.mp_ctx.Queue()
        self.cmd_queue = self.mp_ctx.Queue()

        worker_class = self._get_worker_class()

        for i, device_id in enumerate(self.device_ids):
            # Calculate global rank
            rank = i  # Assuming sequential ranks starting from 0

            # Create ready event for this worker
            ready_event = self.mp_ctx.Event()
            self.ready_events.append(ready_event)

            # Prepare kwargs for worker
            worker_kwargs = self._create_worker_kwargs()
            worker_kwargs.update(
                {
                    "worker_id": i,
                    "device_id": device_id,
                    "rank": rank,
                    "world_size": self.world_size,
                    "init_method": self.init_method,
                }
            )

            # Start worker process
            p = self.mp_ctx.Process(
                target=worker_loop,
                args=(
                    i,
                    device_id,
                    rank,
                    self.world_size,
                    self.init_method,
                    worker_class,
                    worker_kwargs,
                    self.param_queue,
                    self.data_queue,
                    self.loss_queue,
                    self.cmd_queue,
                    ready_event,
                ),
            )
            p.start()
            self.workers.append(p)

        # Wait for all workers to be ready
        for event in self.ready_events:
            event.wait()

    def write_params_to_workers(
        self, trainable_params: Dict[str, Any], blocking: bool = True
    ):
        """
        Write initial parameters to all workers via shared memory.

        Called during Trainer initialization to set up workers with initial model parameters.

        Args:
            trainable_params: Dictionary containing:
                - eval_agent_group: AgentGroup parameters dict
                - target_agent_group: AgentGroup parameters dict
                - eval_critic: Critic state dict
                - target_critic: Target critic state dict
            blocking: Whether to wait for workers to acknowledge
        """
        # Send initial parameters to all workers
        for _ in range(self.world_size):
            self.cmd_queue.put("SYNC_FROM_MAIN")
            self.param_queue.put(trainable_params.copy())

        if blocking:
            # Wait for workers to acknowledge
            for _ in range(self.world_size):
                ack = self.param_queue.get()
                if ack != "ACK":
                    raise RuntimeError(f"Expected ACK from worker, got {ack}")

    def broadcast_params(self):
        """
        Broadcast current Trainer parameters to all workers.

        Called before training starts to ensure workers have latest parameters
        (e.g., after loading from checkpoint).
        """
        # Get latest parameters from worker 0
        self.cmd_queue.put("SYNC_TO_MAIN")
        latest_params = self.param_queue.get()

        # Broadcast to all workers
        for _ in range(self.world_size):
            self.cmd_queue.put("BROADCAST")
            self.param_queue.put(latest_params.copy())

    def read_params_from_worker0(self) -> Dict[str, Any]:
        """
        Read latest parameters from Worker 0.

        Called by Trainer before evaluation or checkpoint saving to get
        the most up-to-date parameters from workers.

        Returns:
            Dictionary containing latest model parameters
        """
        # Signal Worker 0 to write its parameters
        self.cmd_queue.put("SYNC_TO_MAIN")

        # Receive parameters from Worker 0
        params = self.param_queue.get()

        # Other workers also send ACKs
        for _ in range(self.world_size - 1):
            self.param_queue.get()

        return params

    def train_step(self, batch: Dict[str, Any]) -> float:
        """
        Execute one training step across all workers.

        Distributes the batch to workers, which each compute gradients on
        their data slice, then synchronizes via all_reduce.

        Args:
            batch: Full batch from DataLoader

        Returns:
            Average loss across all workers
        """
        # Send train command to all workers
        for _ in range(self.world_size):
            self.cmd_queue.put("TRAIN_STEP")
            self.data_queue.put(batch)

        # Collect losses from all workers
        losses = []
        for _ in range(self.world_size):
            loss = self.loss_queue.get()
            losses.append(loss)

        return sum(losses) / len(losses)

    def shutdown(self):
        """
        Stop all worker processes and clean up resources.
        """
        # Send stop command to all workers
        for _ in range(self.world_size):
            self.cmd_queue.put("STOP")

        # Wait for workers to exit
        for p in self.workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

        self.workers = []
