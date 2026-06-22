"""
QTRAN worker group implementation.

Manages :class:`QTRANWorker` instances for multi-GPU QTRAN training.
The QTRAN-specific knobs (``v_net_config``, ``v_optimizer_config``,
``v_lr_scheduler_config``, ``lambda_opt``, ``lambda_nopt``,
``is_optimal_mask_mode``) are stored on the group and forwarded to
each worker during construction.
"""

from typing import Any, Dict, Optional

from marlite.algorithm.agents import AgentGroupConfig
from marlite.algorithm.critic import CriticConfig
from marlite.algorithm.critic.state_value_config import StateValueConfig
from marlite.util.optimizer_config import OptimizerConfig
from marlite.util.lr_scheduler_config import LRSchedulerConfig
from marlite.trainer.trainer_worker_group.base_worker_group import OffPolicyWorkerGroup


class QTRANWorkerGroup(OffPolicyWorkerGroup):
    """Worker group for QTRAN algorithm multi-GPU training.

    Each worker holds:
    - eval_agent_group + target_agent_group
    - eval_critic + target_critic
    - eval_v_net  (no target V)

    All three trainable models have their own optimizers and are
    step'd independently after ``reduce_gradients`` synchronises
    gradients across workers.
    """

    def __init__(
        self,
        device_ids: list,
        agent_group_config: AgentGroupConfig,
        critic_config: CriticConfig,
        critic_optimizer_config: OptimizerConfig,
        agent_optimizer_config: OptimizerConfig,
        v_net_config: StateValueConfig,
        v_optimizer_config: OptimizerConfig,
        v_lr_scheduler_config: Optional[LRSchedulerConfig] = None,
        gamma: float = 0.95,
        max_grad_norm: float = 5.0,
        lambda_opt: float = 1.0,
        lambda_nopt: float = 1.0,
        is_optimal_mask_mode: bool = True,
        init_method: str = None,
    ):
        self.agent_group_config = agent_group_config
        self.critic_config = critic_config
        self.critic_optimizer_config = critic_optimizer_config
        self.agent_optimizer_config = agent_optimizer_config
        self.v_net_config = v_net_config
        self.v_optimizer_config = v_optimizer_config
        self.v_lr_scheduler_config = v_lr_scheduler_config
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.lambda_opt = lambda_opt
        self.lambda_nopt = lambda_nopt
        self.is_optimal_mask_mode = is_optimal_mask_mode

        super().__init__(
            device_ids=device_ids,
            world_size=len(device_ids),
            init_method=init_method,
        )

    def _get_worker_class(self):
        from marlite.trainer.trainer_worker.qtran_worker import QTRANWorker

        return QTRANWorker

    def _create_worker_kwargs(self) -> Dict[str, Any]:
        """Forward all QTRAN-specific configs and hyperparameters."""
        kwargs = super()._create_worker_kwargs()
        kwargs["gamma"] = self.gamma
        kwargs["max_grad_norm"] = self.max_grad_norm
        kwargs["lambda_opt"] = self.lambda_opt
        kwargs["lambda_nopt"] = self.lambda_nopt
        kwargs["is_optimal_mask_mode"] = self.is_optimal_mask_mode
        kwargs["agent_group_config"] = self.agent_group_config
        kwargs["critic_config"] = self.critic_config
        kwargs["critic_optimizer_config"] = self.critic_optimizer_config
        kwargs["agent_optimizer_config"] = self.agent_optimizer_config
        kwargs["v_net_config"] = self.v_net_config
        kwargs["v_optimizer_config"] = self.v_optimizer_config
        kwargs["v_lr_scheduler_config"] = self.v_lr_scheduler_config
        return kwargs
