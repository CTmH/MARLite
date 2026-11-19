"""torchrun initialization, without algorithm-specific worker processes."""

from datetime import timedelta
import os

import torch
import torch.distributed as dist

from marlite.learning.backend import ExecutionBackend


def initialize_backend(device="auto", timeout_seconds=120):
    """One process/device; torchrun supplies rank and rendezvous environment."""
    if device == "auto":
        device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}" if torch.cuda.is_available() else "cpu"
    target = torch.device(device)
    if target.type == "cuda":
        torch.cuda.set_device(target)
    if "RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group("nccl" if target.type == "cuda" else "gloo",
                                timeout=timedelta(seconds=timeout_seconds))
    return ExecutionBackend(target)
