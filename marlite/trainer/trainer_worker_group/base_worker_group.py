"""
Base worker group module for multi-GPU training.

This module provides the BaseWorkerGroup class that manages multiple worker processes
for parallel training across multiple GPUs.
"""

import io
import socket
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
import torch.multiprocessing as mp


def is_port_available(port: int) -> bool:
    """Check if a port is available for use."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", port))
            return True
    except OSError:
        return False


def serialize_params(params: Dict[str, Any]) -> bytes:
    """
    Serialize parameters to bytes using torch.save.

    This avoids PyTorch's automatic shared memory mechanism which can cause
    file descriptor exhaustion when passing large parameter dictionaries.

    Args:
        params: Dictionary containing parameter data

    Returns:
        Serialized bytes
    """
    buffer = io.BytesIO()
    torch.save(params, buffer)
    return buffer.getvalue()


def deserialize_params(data: bytes) -> Dict[str, Any]:
    """
    Deserialize parameters from bytes using torch.load.

    Args:
        data: Serialized bytes

    Returns:
        Dictionary containing parameter data
    """
    buffer = io.BytesIO(data)
    return torch.load(buffer, weights_only=True)


def _dict_to_cpu(data: Any) -> Any:
    """Recursively convert all tensors in a dict/list to CPU."""
    if isinstance(data, (torch.Tensor, torch.nn.Parameter)):
        if data.is_cuda:
            return data.detach().cpu()
        else:
            return data
    elif hasattr(data, "items"):
        return {k: _dict_to_cpu(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(_dict_to_cpu(x) for x in data)
    return data


def _slice_batch(batch: Dict[str, Any], num_slices: int) -> List[Dict[str, Any]]:
    """
    Slice a batch into multiple sub-batches for data parallelism.

    Args:
        batch: Dictionary containing batch data
        num_slices: Number of slices to create

    Returns:
        List of batch slices
    """
    slices = [{} for _ in range(num_slices)]

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            step = value.shape[0] // num_slices
            for i in range(num_slices):
                slices[i][key] = value[
                    i * step : (i + 1) * step if i < num_slices - 1 else None
                ].clone()
        elif isinstance(value, (list, tuple)):
            # Slice list - divide indices evenly
            step = len(value) // num_slices
            for i in range(num_slices):
                start = i * step
                end = (i + 1) * step if i < num_slices - 1 else len(value)
                slices[i][key] = value[start:end]
        else:
            # Non-sliceable data (scalars, strings, etc.) - keep as is
            for i in range(num_slices):
                slices[i][key] = value

    return slices


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
    ack_queue,
    ready_event,
):
    """
    Main loop function that runs in each worker process.

    Workers wait for commands from the main process and execute them:
    - STOP: Exit the worker loop
    - SYNC_FROM_MAIN: Receive initial parameters from main process
    - BROADCAST: Receive broadcasted parameters from main process
    - SYNC_TO_MAIN: Send current parameters back to main process
    - TRAIN_STEP: Execute one training step on received batch data

    Args:
        worker_id: Unique worker identifier
        device_id: CUDA device ID for this worker
        rank: Global rank in distributed training
        world_size: Total number of processes
        init_method: URL for distributed initialization
        worker_class: Class of the worker to instantiate
        worker_kwargs: Keyword arguments for worker initialization
        param_queue: Queue for parameter exchange
        data_queue: Queue for receiving training data
        loss_queue: Queue for sending loss values back to main process
        cmd_queue: Queue for receiving commands
        ack_queue: Queue for sending ACK signals back to main process
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
            cmd, param_queue, data_queue, loss_queue, ack_queue
        )

        if not should_continue:
            break


class BaseWorkerGroup(ABC):
    """
    Base worker group for managing multiple GPU workers.

    This class handles:
    - Starting and managing worker processes
    - Parameter synchronization via queues (with proper memory isolation)
    - Distributing training batches to workers
    - Collecting loss values from workers

    Subclasses should implement:
    - _create_worker(): Create worker instance with proper model setup
    - _get_worker_class(): Return the worker class to use
    """

    _port_counter = 22100
    _port_lock = threading.Lock()
    _max_port = 65535

    def __init__(
        self,
        device_ids: List[int],
        world_size: int,
        init_method: str = None,
    ):
        """
        Initialize the worker group.

        Args:
            device_ids: List of CUDA device IDs to use
            world_size: Total number of processes (should match len(device_ids))
            init_method: URL for distributed initialization. If None, auto-selects an available port.
        """
        self.device_ids = device_ids
        self.world_size = world_size

        if init_method is None:
            with BaseWorkerGroup._port_lock:
                port = BaseWorkerGroup._port_counter
                while not is_port_available(port):
                    port += 1
                    if port > BaseWorkerGroup._max_port:
                        raise RuntimeError(
                            f"Port counter exceeded maximum port number {BaseWorkerGroup._max_port}"
                        )
                BaseWorkerGroup._port_counter = port + 1
            self.init_method = f"tcp://localhost:{port}"
        else:
            self.init_method = init_method

        # Multiprocessing context
        self.mp_ctx = mp.get_context("spawn")

        # Worker processes and queues
        self.workers = []
        self.loss_queue = None
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

        Note: Using SimpleQueue to avoid automatic shared memory allocation,
        which can cause file descriptor exhaustion with large models.
        """
        self.loss_queue = self.mp_ctx.SimpleQueue()

        # Create separate queues for each worker to avoid race conditions
        self.cmd_queues = []
        self.param_queues = []
        self.data_queues = []
        self.ack_queues = []
        for _ in range(self.world_size):
            self.cmd_queues.append(self.mp_ctx.SimpleQueue())
            self.param_queues.append(self.mp_ctx.SimpleQueue())
            self.data_queues.append(self.mp_ctx.SimpleQueue())
            self.ack_queues.append(self.mp_ctx.SimpleQueue())

        worker_class = self._get_worker_class()

        for i, device_id in enumerate(self.device_ids):
            rank = i
            ready_event = self.mp_ctx.Event()
            self.ready_events.append(ready_event)

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
                    self.param_queues[i],
                    self.data_queues[i],
                    self.loss_queue,
                    self.cmd_queues[i],
                    self.ack_queues[i],
                    ready_event,
                ),
            )
            p.start()
            self.workers.append(p)

        for event in self.ready_events:
            event.wait()

    def write_params_to_workers(
        self, trainable_params: Dict[str, Any], blocking: bool = True
    ):
        """
        Write initial parameters to all workers.

        Called during Trainer initialization to set up workers with initial model parameters.
        Uses serialization to avoid shared memory issues.

        Args:
            trainable_params: Dictionary containing:
                - eval_agent_group: AgentGroup parameters dict
                - target_agent_group: AgentGroup parameters dict
                - eval_critic: Critic state dict
                - target_critic: Target critic state dict
            blocking: Whether to wait for workers to acknowledge
        """
        # Convert to CPU and serialize to bytes
        trainable_params_cpu = _dict_to_cpu(trainable_params)
        serialized_params = serialize_params(trainable_params_cpu)

        for i in range(self.world_size):
            self.cmd_queues[i].put("SYNC_FROM_MAIN")
            # Send serialized bytes - each worker will deserialize independently
            self.param_queues[i].put(serialized_params)

        if blocking:
            for i in range(self.world_size):
                ack = self.ack_queues[i].get()
                if ack != "ACK":
                    raise RuntimeError(f"Worker {i}: Expected ACK, got {ack}")

    def broadcast_params(self, params: Dict[str, Any]):
        """
        Broadcast parameters from trainer to all workers.

        Uses serialization to avoid shared memory issues.

        Args:
            params: Parameters to broadcast.
        """
        params_cpu = _dict_to_cpu(params)
        serialized_params = serialize_params(params_cpu)

        for i in range(self.world_size):
            self.cmd_queues[i].put("BROADCAST")
            self.param_queues[i].put(serialized_params)

    def read_params_from_worker0(self) -> Dict[str, Any]:
        """
        Read latest parameters from Worker 0.

        Called by Trainer before evaluation or checkpoint saving to get
        the most up-to-date parameters from workers.

        Returns:
            Dictionary containing cloned model parameters
        """
        self.cmd_queues[0].put("SYNC_TO_MAIN")
        params = self.param_queues[0].get()
        return _dict_to_cpu(params)

    def read_target_params_from_worker0(self) -> Dict[str, Any]:
        """Read target parameters from Worker 0.

        Workers do per-batch target updates locally (their eval models are
        scattered and target is hard-coupled to that eval).  This method
        pulls worker 0's target state back to the master so the master
        view stays in sync — and the next epoch's broadcast
        (``_sync_params_to_workers``) propagates it to all workers, which
        prevents cross-worker drift.
        """
        self.cmd_queues[0].put("SYNC_TARGET_TO_MAIN")
        params = self.param_queues[0].get()
        return _dict_to_cpu(params)

    def train_step(self, batch: Dict[str, Any]) -> float:
        """
        Execute one training step across all workers.

        Distributes the batch slices to workers, each computes gradients on
        its data slice, then synchronizes via all_reduce.

        Args:
            batch: Full batch from DataLoader

        Returns:
            Average loss across all workers
        """
        batch_slices = _slice_batch(batch, self.world_size)
        for i in range(self.world_size):
            self.cmd_queues[i].put("TRAIN_STEP")
            self.data_queues[i].put(batch_slices[i])

        losses = []
        for _ in range(self.world_size):
            loss = self.loss_queue.get()
            losses.append(loss)

        # No need to sync parameters after each batch because:
        # 1. Gradients are already synchronized via all_reduce in reduce_gradients()
        # 2. All workers use the same optimizer, so parameter updates should be identical
        # 3. Parameters are synced at the beginning of each epoch via broadcast_params()

        return sum(losses) / len(losses)

    def move_models_to_gpu(self):
        """
        Move all workers' models to their assigned GPU devices.
        """
        for i in range(self.world_size):
            self.cmd_queues[i].put("MOVE_TO_GPU")
        for i in range(self.world_size):
            ack = self.ack_queues[i].get()
            if ack != "ACK":
                raise RuntimeError(f"Worker {i}: Expected ACK, got {ack}")

    def move_models_to_cpu(self):
        """
        Move all workers' models to CPU and clear GPU cache.
        """
        for i in range(self.world_size):
            self.cmd_queues[i].put("MOVE_TO_CPU")
        for i in range(self.world_size):
            ack = self.ack_queues[i].get()
            if ack != "ACK":
                raise RuntimeError(f"Worker {i}: Expected ACK, got {ack}")

    def sync_lr_to_workers(self, critic_lr: float, agent_lr: float):
        """
        Synchronize learning rates to all workers.

        Called within _sync_params_to_workers() to ensure workers use the same
        learning rates as the trainer.

        Args:
            critic_lr: Current critic learning rate
            agent_lr: Current agent group learning rate
        """
        lr_data = {"critic_lr": critic_lr, "agent_lr": agent_lr}
        for i in range(self.world_size):
            self.cmd_queues[i].put("SYNC_LR")
            self.param_queues[i].put(lr_data)

        for i in range(self.world_size):
            ack = self.ack_queues[i].get()
            if ack != "ACK":
                raise RuntimeError(f"Worker {i}: Expected ACK, got {ack}")

    def shutdown(self):
        """
        Stop all worker processes and clean up resources.
        """
        for i in range(self.world_size):
            self.cmd_queues[i].put("STOP")

        for p in self.workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

        self.workers = []


class OffPolicyWorkerGroup(BaseWorkerGroup):
    """Base for off-policy worker groups (QMIX, GraphQMIX, etc.)."""


class OnPolicyWorkerGroup(BaseWorkerGroup):
    """Base for on-policy worker groups (MAPPO, G2ANetMAPPO, etc.)."""
