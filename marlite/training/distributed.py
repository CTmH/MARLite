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
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1 and int(os.environ.get("LOCAL_WORLD_SIZE", world)) != world:
        raise ValueError("This execution backend supports single-node torchrun only")
    if world > 1 and target.type not in ("cpu", "cuda"):
        raise ValueError("Distributed execution requires CPU/Gloo or CUDA/ROCm")
    if target.type == "cuda":
        if target.index is None:
            target = torch.device("cuda", local_rank)
        if world > 1 and target.index != local_rank:
            raise ValueError("Use --device cuda/auto under torchrun, one learner per local GPU")
        torch.cuda.set_device(target)
    if "RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group("nccl" if target.type == "cuda" else "gloo",
                                timeout=timedelta(seconds=timeout_seconds))
    return ExecutionBackend(target)
