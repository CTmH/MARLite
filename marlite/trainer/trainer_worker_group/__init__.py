"""
Trainer worker group module for multi-GPU training.

This module provides worker group classes that manage multiple worker processes
for parallel training across multiple GPUs.
"""

from marlite.trainer.trainer_worker_group.base_worker_group import BaseWorkerGroup
from marlite.trainer.trainer_worker_group.qmix_worker_group import QMIXWorkerGroup
from marlite.trainer.trainer_worker_group.graph_worker_group import GraphWorkerGroup
from marlite.trainer.trainer_worker_group.msg_aggr_worker_group import (
    MsgAggrWorkerGroup,
)
from marlite.trainer.trainer_worker_group.ssl_worker_group import SSLWorkerGroup
from marlite.trainer.trainer_worker_group.vae_ssl_worker_group import VAESSLWorkerGroup

__all__ = [
    "BaseWorkerGroup",
    "QMIXWorkerGroup",
    "GraphWorkerGroup",
    "MsgAggrWorkerGroup",
    "SSLWorkerGroup",
    "VAESSLWorkerGroup",
]
