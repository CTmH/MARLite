"""
Distributed Data Parallel (DDP) utilities for multi-GPU training.

This module provides utilities for initializing and managing DDP training,
following the PyTorch Geometric multi-GPU tutorial approach.
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Union, List, Optional


def setup_ddp(
    rank: int,
    world_size: int,
    backend: str = "nccl",
    master_addr: Optional[str] = None,
    master_port: Optional[str] = None,
) -> None:
    """
    Initialize the distributed environment.

    Args:
        rank: Rank of the current process
        world_size: Total number of processes
        backend: Backend to use ('nccl' for GPU, 'gloo' for CPU)
        master_addr: Master address for distributed training (defaults to environment variable or 'localhost')
        master_port: Master port for distributed training (defaults to environment variable or '12355')
    """
    os.environ["MASTER_ADDR"] = (
        master_addr
        if master_addr is not None
        else os.environ.get("MASTER_ADDR", "localhost")
    )
    os.environ["MASTER_PORT"] = (
        master_port
        if master_port is not None
        else os.environ.get("MASTER_PORT", "12355")
    )

    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup_ddp() -> None:
    """Clean up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_device_list(train_device: Union[str, List[str]]) -> tuple[List[str], bool]:
    """
    Parse train_device configuration and return device list.

    Args:
        train_device: Either a string (single device) or a list of strings (multiple devices)
            - "cpu": CPU training
            - "cuda" or "cuda:0": Single GPU training
            - ["cuda:0", "cuda:1", ...]: Multi-GPU training with DDP

    Returns:
        tuple of (device_list, use_ddp):
            - device_list: List of device strings
            - use_ddp: Whether to use DistributedDataParallel
    """
    if isinstance(train_device, str):
        # Single device training (no DDP)
        if train_device == "cuda":
            # Default to cuda:0
            return ["cuda:0"], False
        else:
            return [train_device], False
    elif isinstance(train_device, list):
        # Multi-device training with DDP
        if len(train_device) == 0:
            raise ValueError("train_device list cannot be empty")
        return train_device, True
    else:
        raise ValueError(
            f"train_device must be a string or a list of strings, got {type(train_device)}"
        )


def get_local_device_id(device_str: str) -> int:
    """
    Extract local device ID from device string.

    Args:
        device_str: Device string like 'cuda:0' or 'cpu'

    Returns:
        Local device ID (0 for CPU)
    """
    if device_str.startswith("cuda:"):
        return int(device_str.split(":")[1])
    elif device_str == "cuda":
        return 0
    else:
        return 0  # CPU


def wrap_model_with_ddp(
    model: torch.nn.Module, device_id: int, find_unused_parameters: bool = False
) -> DDP:
    """
    Wrap a model with DistributedDataParallel.

    Args:
        model: The model to wrap
        device_id: The device ID to use
        find_unused_parameters: Whether to find unused parameters (needed for some models)

    Returns:
        DDP wrapped model
    """
    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    ddp_model = DDP(
        model,
        device_ids=[device_id] if device != "cpu" else None,
        output_device=device_id if device != "cpu" else None,
        find_unused_parameters=find_unused_parameters,
    )

    return ddp_model


def unwrap_ddp_model(ddp_model: DDP) -> torch.nn.Module:
    """
    Unwrap a DDP model to get the underlying module.

    Args:
        ddp_model: DDP wrapped model

    Returns:
        The underlying model module
    """
    return ddp_model.module


def is_ddp_model(model: torch.nn.Module) -> bool:
    """
    Check if a model is wrapped with DDP.

    Args:
        model: The model to check

    Returns:
        True if model is a DDP model, False otherwise
    """
    return isinstance(model, DDP)


def reduce_gradients(model: torch.nn.Module, world_size: int) -> None:
    """
    Average gradients across all processes (used for manual gradient synchronization).

    Note: DDP automatically synchronizes gradients during backward pass,
    so this is typically not needed unless doing custom gradient manipulation.

    Args:
        model: The model whose gradients to reduce
        world_size: Total number of processes
    """
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= world_size


def gather_tensors(tensor: torch.Tensor, world_size: int) -> List[torch.Tensor]:
    """
    Gather tensors from all processes to all processes.

    Args:
        tensor: Local tensor
        world_size: Total number of processes

    Returns:
        List of tensors from all processes
    """
    if world_size == 1:
        return [tensor]

    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    return tensor_list


def average_loss(loss: torch.Tensor, world_size: int) -> torch.Tensor:
    """
    Average loss across all processes.

    Args:
        loss: Local loss tensor
        world_size: Total number of processes

    Returns:
        Averaged loss tensor
    """
    if world_size == 1:
        return loss

    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
    loss /= world_size
    return loss
