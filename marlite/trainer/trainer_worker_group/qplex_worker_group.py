"""QPLEX worker group — manages QPLEX workers for multi-GPU training.

Mirrors :class:`QMIXWorkerGroup`.  Each worker owns a local copy of
the eval/target agent groups and critics.  The group is responsible for
starting the worker processes, broadcasting model parameters, and
collecting training results.
"""

from typing import Any, Dict

from marlite.trainer.trainer_worker_group.base_worker_group import OffPolicyWorkerGroup
from marlite.trainer.trainer_worker.qplex_worker import QPLEXWorker


class QPLEXWorkerGroup(OffPolicyWorkerGroup):
    """Worker group for QPLEX multi-GPU training.

    Args:
        device_ids: List of CUDA device IDs to use.
        agent_group_config: Configuration for the agent group.
        critic_config: Configuration for the QPLEXMixer.
        critic_optimizer_config: Configuration for the critic optimiser.
        agent_optimizer_config: Configuration for the agent optimiser.
        gamma: Discount factor.
        max_grad_norm: Maximum gradient norm for clipping.
        init_method: URL for distributed initialisation.
    """

    def __init__(
        self,
        device_ids: list,
        agent_group_config,
        critic_config,
        critic_optimizer_config,
        agent_optimizer_config,
        gamma: float = 0.9,
        max_grad_norm: float = 5.0,
        init_method: str = None,
    ):
        self.agent_group_config = agent_group_config
        self.critic_config = critic_config
        self.critic_optimizer_config = critic_optimizer_config
        self.agent_optimizer_config = agent_optimizer_config
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm

        super().__init__(
            device_ids=device_ids,
            world_size=len(device_ids),
            init_method=init_method,
        )

    def _get_worker_class(self):
        """Return the :class:`QPLEXWorker` class for instantiation."""
        return QPLEXWorker

    def _create_worker_kwargs(self) -> Dict[str, Any]:
        """Build the keyword arguments dict for each worker's constructor."""
        kwargs = super()._create_worker_kwargs()
        kwargs["gamma"] = self.gamma
        kwargs["max_grad_norm"] = self.max_grad_norm
        kwargs["agent_group_config"] = self.agent_group_config
        kwargs["critic_config"] = self.critic_config
        kwargs["critic_optimizer_config"] = self.critic_optimizer_config
        kwargs["agent_optimizer_config"] = self.agent_optimizer_config
        return kwargs

    def set_worker_models(self):
        """No-op — workers create their own model copies in their constructor."""
        pass
