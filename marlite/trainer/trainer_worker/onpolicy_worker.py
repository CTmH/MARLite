"""OnPolicyWorker — worker for on-policy algorithms (MAPPO, etc.).

No target-agent or target-critic — they are not needed for on-policy
methods.  This class owns the standard critic + agent learning-rate
sync via ``handle_command("SYNC_LR")``; algorithm-specific extras
(e.g. ``ssl_optimizer`` for self-supervised on-policy variants) are
owned by further subclasses.
"""

from typing import Any, Dict
import torch
import torch.distributed as dist
from marlite.trainer.trainer_worker.base_worker import BaseWorker


class OnPolicyWorker(BaseWorker):
    """Worker for on-policy algorithms (MAPPO, G2ANetMAPPO, …).

    Does NOT hold target networks (``target_agent_group`` /
    ``target_critic``) since on-policy methods do not use them.
    Handles ``SYNC_LR`` for the standard critic and agent optimizers.
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

    def reduce_gradients(self):
        """All-reduce gradients across all workers for ``eval_critic``
        and ``eval_agent_group``.  Subclasses that own additional
        modules (e.g. ``ssl_model``) override this and call
        ``super().reduce_gradients()`` first.
        """
        for net in (self.eval_critic, self.eval_agent_group):
            for param in net.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= self.world_size

    def handle_command(
        self, cmd, param_queue, data_queue, loss_queue, ack_queue=None
    ) -> bool:
        if cmd == "SYNC_LR":
            lr_data = param_queue.get()
            if "critic_lr" in lr_data and self.critic_optimizer is not None:
                for pg in self.critic_optimizer.param_groups:
                    pg["lr"] = lr_data["critic_lr"]
            if "agent_lr" in lr_data and self.agent_optimizer is not None:
                for pg in self.agent_optimizer.param_groups:
                    pg["lr"] = lr_data["agent_lr"]
            if ack_queue:
                ack_queue.put("ACK")
            return True
        return super().handle_command(cmd, param_queue, data_queue, loss_queue, ack_queue)
