"""OffPolicyWorker — adds target-agent and target-critic for off-policy RL."""

from typing import Any, Dict
from marlite.trainer.trainer_worker.base_worker import BaseWorker


class OffPolicyWorker(BaseWorker):
    """Worker for off-policy algorithms (QMIX, GraphQMIX, etc.).

    Extends :class:`BaseWorker` with:
    - target_agent_group  (periodic copy of eval_agent_group)
    - target_critic       (periodic copy of eval_critic)
    - get_params_for_main that includes target models
    """

    def __init__(
        self,
        worker_id: int,
        device_id: int,
        rank: int,
        world_size: int,
        init_method: str,
    ):
        super().__init__(worker_id, device_id, rank, world_size, init_method)
        self.target_agent_group = None
        self.target_critic = None

    def get_params_for_main(self) -> Dict[str, Any]:
        params = {
            "eval_agent_group": {
                k: v.clone().cpu()
                for k, v in self.eval_agent_group.state_dict().items()
            },
            "eval_critic": {
                k: v.clone().cpu() for k, v in self.eval_critic.state_dict().items()
            },
        }
        if self.target_agent_group is not None:
            params["target_agent_group"] = {
                k: v.clone().cpu()
                for k, v in self.target_agent_group.state_dict().items()
            }
        if self.target_critic is not None:
            params["target_critic"] = {
                k: v.clone().cpu()
                for k, v in self.target_critic.state_dict().items()
            }
        return params