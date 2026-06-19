"""
Clean base worker with no target-network assumptions.

Subclasses inject target models when needed (off-policy) or omit
them (on-policy).  Must override :meth:`get_params_for_main`.
"""

import io
import torch
import torch.distributed as dist
from typing import Any, Dict, Optional


class BaseWorker:
    """Minimal multi-GPU worker base.

    Each worker runs in a separate process and may hold copies of:
    - eval_agent_group
    - eval_critic
    Subclasses for off-policy training add target_agent_group / target_critic.
    """

    def __init__(
        self,
        worker_id: int,
        device_id: int,
        rank: int,
        world_size: int,
        init_method: str,
    ):
        self.worker_id = worker_id
        self.device_id = device_id
        self.rank = rank
        self.world_size = world_size
        self.init_method = init_method

        self.assigned_device = f"cuda:{device_id}"
        self.device = "cpu"
        self._setup_distributed()

        self.eval_agent_group = None
        self.eval_critic = None
        self.critic_optimizer: Optional[torch.optim.Optimizer] = None
        self.agent_optimizer: Optional[torch.optim.Optimizer] = None
        self.ssl_optimizer: Optional[torch.optim.Optimizer] = None

    def _setup_distributed(self):
        torch.cuda.set_device(self.device_id)
        dist.init_process_group(
            backend="nccl",
            init_method=self.init_method,
            rank=self.rank,
            world_size=self.world_size,
        )

    # ------------------------------------------------------------------
    # Parameter sync — guards protect optional submodules (target, ssl)
    # ------------------------------------------------------------------

    def sync_params_from_main(self, params):
        if isinstance(params, bytes):
            buf = io.BytesIO(params)
            params = torch.load(buf, weights_only=True)

        if "eval_agent_group" in params and self.eval_agent_group is not None:
            self.eval_agent_group.load_state_dict(
                {k: v.clone() for k, v in params["eval_agent_group"].items()}
            )
        if "eval_critic" in params and self.eval_critic is not None:
            self.eval_critic.load_state_dict(
                {k: v.clone() for k, v in params["eval_critic"].items()}
            )

        # Off-policy subclasses may add these; guard via hasattr + None.
        _load_opt(self, "target_agent_group", params)
        _load_opt(self, "target_critic", params)
        _load_opt(self, "ssl_model", params)

        if "reward_aggr_mode" in params:
            self.reward_aggr_mode = params["reward_aggr_mode"]

    def _aggregate_rewards(self, rewards: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Aggregate per-agent rewards; see ``Trainer._aggregate_rewards``."""
        mode = getattr(self, "reward_aggr_mode", "sum")
        if mode == "sum":
            return rewards.sum(dim=dim)
        elif mode == "mean":
            return rewards.mean(dim=dim)
        else:
            raise ValueError(f"Unknown reward_aggr_mode '{mode}'")

    def move_to_device(self, device: str):
        if self.eval_agent_group is not None:
            self.eval_agent_group.to(device)
        if self.eval_critic is not None:
            self.eval_critic.to(device)
        _to_opt(self, "target_agent_group", device)
        _to_opt(self, "target_critic", device)
        _to_opt(self, "ssl_model", device)
        self.device = device

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    def get_params_for_main(self) -> Dict[str, Any]:
        raise NotImplementedError("subclass must implement get_params_for_main")

    def reduce_gradients(self):
        for param in self.eval_critic.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.world_size
        for param in self.eval_agent_group.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.world_size

    def train_step(self, batch: Dict[str, Any]) -> float:
        raise NotImplementedError("subclasses must implement train_step()")

    def handle_command(
        self, cmd, param_queue, data_queue, loss_queue, ack_queue=None
    ) -> bool:
        if cmd == "STOP":
            self.cleanup()
            return False
        elif cmd == "SYNC_FROM_MAIN":
            params = param_queue.get()
            self.sync_params_from_main(params)
            del params
            ack_queue.put("ACK")
        elif cmd == "BROADCAST":
            params = param_queue.get()
            self.sync_params_from_main(params)
            del params
        elif cmd == "SYNC_TO_MAIN":
            params = self.get_params_for_main()
            param_queue.put(params)
        elif cmd == "TRAIN_STEP":
            batch = data_queue.get()
            loss = self.train_step(batch)
            del batch
            loss_queue.put(loss)
        elif cmd == "MOVE_TO_GPU":
            self.move_to_device(self.assigned_device)
            if ack_queue:
                ack_queue.put("ACK")
        elif cmd == "MOVE_TO_CPU":
            self.move_to_device("cpu")
            torch.cuda.empty_cache()
            if ack_queue:
                ack_queue.put("ACK")
        elif cmd == "SYNC_LR":
            lr_data = param_queue.get()
            if "critic_lr" in lr_data:
                for pg in self.critic_optimizer.param_groups:
                    pg["lr"] = lr_data["critic_lr"]
            if "agent_lr" in lr_data:
                for pg in self.agent_optimizer.param_groups:
                    pg["lr"] = lr_data["agent_lr"]
            if hasattr(self, "ssl_optimizer") and self.ssl_optimizer is not None:
                if "ssl_lr" in lr_data:
                    for pg in self.ssl_optimizer.param_groups:
                        pg["lr"] = lr_data["ssl_lr"]
            if ack_queue:
                ack_queue.put("ACK")
        else:
            print(
                f"Worker {self.worker_id}: Unknown command: {repr(cmd)}",
                flush=True,
            )
        return True

    def cleanup(self):
        if dist.is_initialized():
            dist.destroy_process_group()


# -- helpers that degrade gracefully when the attribute is missing ----

def _load_opt(obj, attr, params):
    if hasattr(obj, attr) and getattr(obj, attr) is not None:
        getattr(obj, attr).load_state_dict(
            {k: v.clone() for k, v in params[attr].items()}
        )


def _to_opt(obj, attr, device):
    if hasattr(obj, attr) and getattr(obj, attr) is not None:
        getattr(obj, attr).to(device)