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
from marlite.trainer.trainer_worker_group.vae_ssl_worker_group import (
    VAEGraphWorkerGroup,
)
from marlite.trainer.trainer_worker_group.group_consensus_worker_group import (
    GroupConsensusWorkerGroup,
)
from marlite.trainer.trainer_worker_group.vae_group_consensus_worker_group import (
    VAEGroupConsensusWorkerGroup,
)

__all__ = [
    "BaseWorkerGroup",
    "QMIXWorkerGroup",
    "GraphWorkerGroup",
    "MsgAggrWorkerGroup",
    "VAEGraphWorkerGroup",
    "GroupConsensusWorkerGroup",
    "VAEGroupConsensusWorkerGroup",
]
