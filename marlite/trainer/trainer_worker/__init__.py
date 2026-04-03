"""
Trainer worker module for multi-GPU training.

This module provides worker classes that run in separate processes to enable
parallel training across multiple GPUs. Each worker holds model copies and
implements training logic specific to its algorithm.
"""

from marlite.trainer.trainer_worker.base_worker import BaseWorker
from marlite.trainer.trainer_worker.qmix_worker import QMIXWorker
from marlite.trainer.trainer_worker.graph_worker import GraphWorker
from marlite.trainer.trainer_worker.vae_graph_worker import VAEGraphQMIXWorker
from marlite.trainer.trainer_worker.msg_aggr_worker import (
    MsgAggrWorker,
    ProbMsgAggrWorker,
)

__all__ = [
    "BaseWorker",
    "QMIXWorker",
    "GraphWorker",
    "VAEGraphQMIXWorker",
    "MsgAggrWorker",
    "ProbMsgAggrWorker",
]
