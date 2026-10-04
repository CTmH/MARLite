"""
Base worker group module for multi-GPU training.

This module provides the BaseWorkerGroup class that manages multiple worker processes
for parallel training across multiple GPUs.
"""

import io
import socket
import threading
import traceback
from queue import Empty
from absl import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
import torch.multiprocessing as mp
from marlite.util.randomness import configure_randomness, derive_seed, seed_everything, WORKER_STREAM


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
    Split samples as evenly as possible (shard sizes differ by at most one).

    Workers still average rank-local losses/gradients equally, not by sample
    count. For B >= D and B = q*D + r samples over D ranks, this changes the
    sample-weight distribution by total variation r*(D-r)/(D*B). This is a small approximation
    when each rank has many samples, not a bound on relative gradient error.

    If B < D, replicate the entire tiny batch on every rank. Averaging identical
    batch objectives preserves sample weights and avoids empty-rank collectives;
    stochastic layers and rank-local statistics can still differ between ranks.

    Args:
        batch: Dictionary containing batch data
        num_slices: Number of slices to create

    Returns:
        List of batch slices
    """
    if num_slices < 1:
        raise ValueError("num_slices must be positive")
    # Validate before dispatch: an empty rank can strand peers in all-reduce.
    sizes = {key: len(value) for key, value in batch.items()
             if isinstance(value, (torch.Tensor, list, tuple)) and
             (not isinstance(value, torch.Tensor) or value.ndim > 0)}
    if not sizes or len(set(sizes.values())) != 1:
        raise ValueError(f"Batch fields must have one common sample count: {sizes}")
    size = next(iter(sizes.values()))
    if size == 0:
        raise ValueError(
            "Batch size must be positive; cannot dispatch an empty batch."
        )
    if size < num_slices:
        bounds = [(0, size)] * num_slices
    else:
        step, remainder = divmod(size, num_slices)
        bounds = []
        start = 0
        for i in range(num_slices):
            end = start + step + (i < remainder)
            bounds.append((start, end))
            start = end
    slices = [{} for _ in range(num_slices)]

    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            for i, (start, end) in enumerate(bounds):
                slices[i][key] = value[start:end].clone()
        elif isinstance(value, (list, tuple)):
            for i, (start, end) in enumerate(bounds):
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
    seed=None,
    deterministic=False,
    error_queue=None,
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
    cmd = "INITIALIZE"
    try:
        worker_seed = derive_seed(seed, WORKER_STREAM, rank)
        configure_randomness(worker_seed, deterministic)
        worker = worker_class(**worker_kwargs)
        seed_everything(worker_seed)
        ready_event.set()
        while True:
            cmd = cmd_queue.get()
            if not worker.handle_command(
                cmd, param_queue, data_queue, loss_queue, ack_queue
            ):
                break
    except BaseException:
        if error_queue is not None:
            error_queue.put(
                f"Worker {worker_id} (device {device_id}), command {cmd!r}:\n"
                f"{traceback.format_exc()}"
            )
            # Flush the traceback before exit. SIGKILL/abort cannot be caught;
            # the parent detects those separately through process exit codes.
            error_queue.close()
            error_queue.join_thread()
        raise


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
        self.error_queue = None

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

        Timed Queue reads let the parent detect worker failures. Parameter
        broadcasts remain serialized bytes to avoid tensor IPC overhead.
        """
        self.loss_queue = self.mp_ctx.Queue()
        self.error_queue = self.mp_ctx.Queue()

        # Create separate queues for each worker to avoid race conditions
        self.cmd_queues = []
        self.param_queues = []
        self.data_queues = []
        self.ack_queues = []
        for _ in range(self.world_size):
            self.cmd_queues.append(self.mp_ctx.Queue())
            self.param_queues.append(self.mp_ctx.Queue())
            self.data_queues.append(self.mp_ctx.Queue())
            self.ack_queues.append(self.mp_ctx.Queue())

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
                    getattr(self, "seed", None),
                    getattr(self, "deterministic", False),
                    self.error_queue,
                ),
            )
            p.start()
            self.workers.append(p)

        for event in self.ready_events:
            while not event.wait(timeout=0.2):
                self._check_workers()
        self._check_workers()

    def _check_workers(self):
        """Raise in the training thread, with a logged traceback or exit code."""
        try:
            error = self.error_queue.get_nowait()
        except Empty:
            dead = [p for p in self.workers if p.exitcode is not None]
            if not dead:
                return
            try:
                error = self.error_queue.get(timeout=0.2)
            except Empty:
                error = "Worker exited unexpectedly: " + ", ".join(
                    f"pid={p.pid}, exitcode={p.exitcode}" for p in dead
                )
        logging.error("Distributed training aborted: %s", error)
        self.shutdown(force=True)
        raise RuntimeError(f"Distributed training aborted: {error}")

    def _receive(self, queue):
        """Wait for a response without waiting forever for a dead worker."""
        while True:
            self._check_workers()
            try:
                return queue.get(timeout=0.2)
            except Empty:
                continue
            except (EOFError, OSError, RuntimeError) as error:
                # A worker can die while a tensor payload is being unpickled.
                self._check_workers()
                logging.exception("Distributed worker response failed")
                self.shutdown(force=True)
                raise RuntimeError("Distributed worker response failed") from error

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
                ack = self._receive(self.ack_queues[i])
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
        params = self._receive(self.param_queues[0])
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
        params = self._receive(self.param_queues[0])
        return _dict_to_cpu(params)

    def average_eval_params(self):
        """Synchronise (all-reduce) eval-parameters across all workers.

        Every worker's ``eval_agent_group``, ``eval_critic`` (and
        algorithm-specific auxiliary add-ons such as ``ssl_model``) are
        averaged so that a subsequent read from worker 0 returns the
        consensus state rather than any single worker's local state.

        Blocks until all workers acknowledge.
        """
        for i in range(self.world_size):
            self.cmd_queues[i].put("AVERAGE_EVAL_PARAMS")
        for i in range(self.world_size):
            ack = self._receive(self.ack_queues[i])
            if ack != "ACK":
                raise RuntimeError(f"Worker {i}: Expected ACK, got {ack}")

    def average_target_params(self):
        """Synchronise (all-reduce) target-parameters across all workers.

        Every worker's ``target_agent_group`` and ``target_critic`` are
        averaged so that the next read from worker 0 returns the
        consensus state.  Prevents per-worker drift in ema trajectories.

        Blocks until all workers acknowledge.
        """
        for i in range(self.world_size):
            self.cmd_queues[i].put("AVERAGE_TARGET_PARAMS")
        for i in range(self.world_size):
            ack = self._receive(self.ack_queues[i])
            if ack != "ACK":
                raise RuntimeError(f"Worker {i}: Expected ACK, got {ack}")

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Execute one training step across all workers.

        Distributes the batch slices to workers, each computes gradients on
        its data slice, then synchronizes via all_reduce.

        Args:
            batch: Full batch from DataLoader

        Returns:
            Per-metric averages across all workers.
        """
        try:
            batch_slices = _slice_batch(batch, self.world_size)
        except ValueError:
            # Non-daemon workers must not keep Python alive after a rejected
            # batch raises out of the training entry point.
            if getattr(self, "workers", None):
                logging.exception("Distributed training rejected an invalid batch")
                self.shutdown(force=True)
            raise
        for i in range(self.world_size):
            self.cmd_queues[i].put("TRAIN_STEP")
            self.data_queues[i].put(batch_slices[i])

        results = []
        for _ in range(self.world_size):
            result = self._receive(self.loss_queue)
            if not isinstance(result, dict):
                raise TypeError(
                    "Worker train_step must return a dict, got "
                    f"{type(result).__name__}"
                )
            results.append(result)

        # No need to sync parameters after each batch because:
        # 1. Gradients are already synchronized via all_reduce in reduce_gradients()
        # 2. All workers use the same optimizer, so parameter updates should be identical
        # 3. Parameters are synced at the beginning of each epoch via broadcast_params()

        keys = results[0].keys()
        if any(result.keys() != keys for result in results[1:]):
            raise ValueError("Workers returned different training metric keys")
        return {
            key: sum(result[key] for result in results) / len(results)
            for key in keys
        }

    def move_models_to_gpu(self):
        """
        Move all workers' models to their assigned GPU devices.
        """
        for i in range(self.world_size):
            self.cmd_queues[i].put("MOVE_TO_GPU")
        for i in range(self.world_size):
            ack = self._receive(self.ack_queues[i])
            if ack != "ACK":
                raise RuntimeError(f"Worker {i}: Expected ACK, got {ack}")

    def move_models_to_cpu(self):
        """
        Move all workers' models to CPU and clear GPU cache.
        """
        for i in range(self.world_size):
            self.cmd_queues[i].put("MOVE_TO_CPU")
        for i in range(self.world_size):
            ack = self._receive(self.ack_queues[i])
            if ack != "ACK":
                raise RuntimeError(f"Worker {i}: Expected ACK, got {ack}")

    def sync_lr_to_workers(
        self, critic_lr: float, agent_lr: float, **extra
    ):
        """
        Synchronize learning rates to all workers.

        Called within ``_sync_params_to_workers`` to ensure workers use the
        same learning rates as the trainer.  ``critic_lr`` and ``agent_lr``
        are always sent; additional algorithm-specific rates are forwarded
        through ``**extra`` and consumed by the corresponding worker's
        ``handle_command("SYNC_LR")`` override (e.g. ``v_lr`` for QTRAN,
        ``ssl_lr`` for self-supervised variants).

        Args:
            critic_lr: Current critic learning rate.
            agent_lr: Current agent group learning rate.
            **extra: Additional learning rates (e.g. ``v_lr=...``,
                ``ssl_lr=...``).  Each entry is included verbatim in the
                per-worker ``lr_data`` payload.
        """
        lr_data = {"critic_lr": critic_lr, "agent_lr": agent_lr, **extra}
        for i in range(self.world_size):
            self.cmd_queues[i].put("SYNC_LR")
            self.param_queues[i].put(lr_data)

        for i in range(self.world_size):
            ack = self._receive(self.ack_queues[i])
            if ack != "ACK":
                raise RuntimeError(f"Worker {i}: Expected ACK, got {ack}")

    def shutdown(self, force=False):
        """
        Stop all worker processes and clean up resources.
        """
        for i, process in enumerate(self.workers):
            if process.is_alive():
                if force:
                    process.terminate()
                else:
                    self.cmd_queues[i].put("STOP")

        for p in self.workers:
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
                p.join(timeout=5)

        self.workers = []
        for queue in [self.loss_queue, self.error_queue,
                      *getattr(self, "cmd_queues", []),
                      *getattr(self, "param_queues", []),
                      *getattr(self, "data_queues", []),
                      *getattr(self, "ack_queues", [])]:
            if queue is not None:
                # Pending payloads may have no reader after a worker failure.
                queue.cancel_join_thread()
                queue.close()


class OffPolicyWorkerGroup(BaseWorkerGroup):
    """Base for off-policy worker groups (QMIX, GraphQMIX, etc.)."""


class OnPolicyWorkerGroup(BaseWorkerGroup):
    """Base for on-policy worker groups (MAPPO, G2ANetMAPPO, etc.)."""
