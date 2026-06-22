"""
Minimal generic worker base.

Subclasses own their own state and the sync semantics for any
algorithm-specific models (target networks, SSL models, V_net, etc.).
``BaseWorker`` itself only knows about ``eval_agent_group``,
``eval_critic`` and ``reward_aggr_mode`` — the parameters that are
present in every trainer.
"""

import io
import torch
import torch.distributed as dist
from typing import Any, Dict, Optional


class BaseWorker:
    """Generic multi-GPU worker base.

    Each worker runs in a separate process and may hold copies of:
    - ``eval_agent_group``
    - ``eval_critic``

    Subclasses for off-policy training add target networks
    (``OffPolicyWorker``); subclasses for self-supervised training add an
    ``ssl_model`` and its optimizer; subclasses for QTRAN add an
    auxiliary ``eval_v_net`` and ``v_optimizer``.  None of that lives in
    this class — each subclass overrides the relevant sync / move
    hooks explicitly.
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
        self.reward_aggr_mode: str = "sum"

    def _setup_distributed(self):
        torch.cuda.set_device(self.device_id)
        dist.init_process_group(
            backend="nccl",
            init_method=self.init_method,
            rank=self.rank,
            world_size=self.world_size,
        )

    # ------------------------------------------------------------------
    # Parameter sync — generic subset only.
    #
    # Algorithm-specific keys (``target_*``, ``target_update_*``,
    # ``ssl_model``, ``eval_v_net``, …) are NOT loaded here.  Subclasses
    # override ``sync_params_from_main`` and add their own branches.
    # ------------------------------------------------------------------

    def sync_params_from_main(self, params):
        """Load the generic per-epoch parameters from the master.

        Accepts either a ``bytes`` blob (serialised via
        :func:`serialize_params`) or a plain ``dict`` of state dicts.
        Returns the deserialised ``dict`` so that subclasses extending
        this method can keep operating on the resolved mapping.
        Subclasses that override this method **must** call
        ``super().sync_params_from_main(params)`` first and use the
        returned ``params`` for their own key lookups.

        Args:
            params: Either a ``bytes`` blob or a ``dict``.

        Returns:
            The deserialised parameter dict.
        """
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

        if "reward_aggr_mode" in params:
            self.reward_aggr_mode = params["reward_aggr_mode"]

        return params

    def get_target_params(self) -> Dict[str, Any]:
        """Return target-network state dicts.

        Default implementation returns an empty dict — no targets.
        ``OffPolicyWorker`` overrides this to include
        ``target_agent_group`` and ``target_critic``.
        """
        return {}

    # ------------------------------------------------------------------
    # Reward aggregation — generic helper shared by all off-policy
    # subclasses.  The mode string is pushed by the master every epoch
    # (see ``sync_params_from_main`` above).
    # ------------------------------------------------------------------

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
        """Move the generic eval models to ``device``.

        Subclasses that own additional modules (``target_*``, ``ssl_model``,
        ``eval_v_net``) override this and call ``super().move_to_device()``
        first.
        """
        if self.eval_agent_group is not None:
            self.eval_agent_group.to(device)
        if self.eval_critic is not None:
            self.eval_critic.to(device)
        self.device = device

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    def get_params_for_main(self) -> Dict[str, Any]:
        raise NotImplementedError("subclass must implement get_params_for_main")

    def reduce_gradients(self):
        """All-reduce gradients across all workers.  Subclasses override."""
        raise NotImplementedError("subclass must implement reduce_gradients")

    def train_step(self, batch: Dict[str, Any]) -> float:
        raise NotImplementedError("subclasses must implement train_step()")

    def handle_command(
        self, cmd, param_queue, data_queue, loss_queue, ack_queue=None
    ) -> bool:
        """Dispatch generic worker commands.

        Subclasses (e.g. ``OffPolicyWorker``, ``OnPolicyWorker``,
        ``QTRANWorker``) extend this with algorithm-specific commands such
        as ``SYNC_LR``.
        """
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
        elif cmd == "SYNC_TARGET_TO_MAIN":
            params = self.get_target_params()
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
        else:
            print(
                f"Worker {self.worker_id}: Unknown command: {repr(cmd)}",
                flush=True,
            )
        return True

    def cleanup(self):
        if dist.is_initialized():
            dist.destroy_process_group()
