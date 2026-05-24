"""OnPolicyWorker — worker for on-policy algorithms (MAPPO, etc.).

No target-agent or target-critic — they are not needed for on-policy methods.
"""

from typing import Any, Dict
from marlite.trainer.trainer_worker.base_worker import BaseWorker


class OnPolicyWorker(BaseWorker):
    """Worker for on-policy algorithms (MAPPO, G2ANetMAPPO, etc.).

    Does NOT hold target networks (target_agent_group / target_critic)
    since on-policy methods do not use them.
    """

    def get_params_for_main(self) -> Dict[str, Any]:
        return {
            "eval_agent_group": {
                k: v.clone().cpu()
                for k, v in self.eval_agent_group.state_dict().items()
            },
            "eval_critic": {
                k: v.clone().cpu() for k, v in self.eval_critic.state_dict().items()
            },
        }