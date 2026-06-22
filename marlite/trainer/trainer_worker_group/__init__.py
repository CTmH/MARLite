"""
Trainer worker group module for multi-GPU training.

This module provides worker group classes that manage multiple worker processes
for parallel training across multiple GPUs.
"""

from marlite.trainer.trainer_worker_group.base_worker_group import (
    BaseWorkerGroup,
    OffPolicyWorkerGroup,
    OnPolicyWorkerGroup,
)
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
from marlite.trainer.trainer_worker_group.ssl_group_consensus_worker_group import (
    SSLGroupConsensusWorkerGroup,
)
from marlite.trainer.trainer_worker_group.qtran_worker_group import QTRANWorkerGroup
from marlite.trainer.trainer_worker_group.mappo_worker_group import (
    MAPPOWorkerGroup,
)
from marlite.trainer.trainer_worker_group.ssl_gc_mappo_worker_group import (
    SSLGroupConsensusMAPPOWorkerGroup,
)

__all__ = [
    "BaseWorkerGroup",
    "OffPolicyWorkerGroup",
    "OnPolicyWorkerGroup",
    "QMIXWorkerGroup",
    "GraphWorkerGroup",
    "MsgAggrWorkerGroup",
    "VAEGraphWorkerGroup",
    "GroupConsensusWorkerGroup",
    "SSLGroupConsensusWorkerGroup",
    "QTRANWorkerGroup",
    "MAPPOWorkerGroup",
    "SSLGroupConsensusMAPPOWorkerGroup",
]
