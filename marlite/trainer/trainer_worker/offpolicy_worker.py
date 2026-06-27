"""OffPolicyWorker — adds target-agent and target-critic for off-policy RL.

This class owns the target-network state and target-update logic; it
handles the per-epoch ``SYNC_LR`` command for the standard critic and
agent optimizers.  Algorithm-specific extras (e.g. ``v_optimizer`` for
QTRAN, ``ssl_optimizer`` for self-supervised variants) are owned by
further subclasses.
"""

from typing import Any, Dict
import torch
import torch.distributed as dist
from marlite.trainer.trainer_worker.base_worker import BaseWorker


class OffPolicyWorker(BaseWorker):
    """Worker for off-policy algorithms (QMIX, GraphQMIX, QTRAN, …).

    Extends :class:`BaseWorker` with:
    - ``target_agent_group``  (periodic copy of ``eval_agent_group``)
    - ``target_critic``       (periodic copy of ``eval_critic``)
    - target-update configuration (``target_update_mode``,
      ``target_update_tau``, ``target_update_interval``)
    - per-batch target refresh via :meth:`_update_target_after_batch`
    - a ``SYNC_LR`` handler that sets ``critic_lr`` and ``agent_lr`` on
      the standard optimizers; subclasses extend it for additional LRs.
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

        # Target-update configuration (pushed by master via SYNC_FROM_MAIN)
        self.target_update_mode: str = "hard"
        self.target_update_tau: float = 0.005
        self.target_update_interval: int = 1
        self._total_batches_processed: int = 0

    # ------------------------------------------------------------------
    # Sync — target / target-update config
    # ------------------------------------------------------------------

    def sync_params_from_main(self, params):
        params = super().sync_params_from_main(params)
        if "target_agent_group" in params and self.target_agent_group is not None:
            self.target_agent_group.load_state_dict(
                {k: v.clone() for k, v in params["target_agent_group"].items()}
            )
        if "target_critic" in params and self.target_critic is not None:
            self.target_critic.load_state_dict(
                {k: v.clone() for k, v in params["target_critic"].items()}
            )
        if "target_update_mode" in params:
            self.target_update_mode = params["target_update_mode"]
        if "target_update_tau" in params:
            self.target_update_tau = float(params["target_update_tau"])
        if "target_update_interval" in params:
            self.target_update_interval = int(params["target_update_interval"])
        return params

    def get_target_params(self) -> Dict[str, Any]:
        """Return current target-network state dicts (used by SYNC_TARGET_TO_MAIN)."""
        params: Dict[str, Any] = {}
        if self.target_agent_group is not None:
            params["target_agent_group"] = {
                k: v.clone() for k, v in self.target_agent_group.state_dict().items()
            }
        if self.target_critic is not None:
            params["target_critic"] = {
                k: v.clone() for k, v in self.target_critic.state_dict().items()
            }
        return params

    # ------------------------------------------------------------------
    # Device move — include target networks
    # ------------------------------------------------------------------

    def move_to_device(self, device: str):
        super().move_to_device(device)
        if self.target_agent_group is not None:
            self.target_agent_group.to(device)
        if self.target_critic is not None:
            self.target_critic.to(device)

    # ------------------------------------------------------------------
    # Target-update logic
    # ------------------------------------------------------------------

    @staticmethod
    def _ema_update(target, source, tau):
        """Polyak averaging: θ_target = τ·θ_source + (1-τ)·θ_target."""
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.mul_(1 - tau).add_(sp.data, alpha=tau)

    def _update_target_after_batch(self):
        """Refresh the target networks on this worker after one gradient batch.

        Mirrors ``OffPolicyTrainer._update_target_after_batch`` so that each
        worker maintains its own copy of the target networks without
        round-tripping through the master on every batch.  Called from
        ``train_step`` immediately after ``optimizer.step()``.

        The local batch counter ``_total_batches_processed`` is incremented
        first; the actual update is then gated by
        ``_total_batches_processed % target_update_interval == 0`` so that
        ``target_update_interval`` throttles all three modes uniformly:

        - ``"hard"``: hard-copy eval-network parameters into target networks.
        - ``"ema"`` / ``"polyak"``: Polyak averaging
          ``θ_target ← τ·θ_eval + (1-τ)·θ_target`` with
          ``τ = target_update_tau``.
        """
        self._total_batches_processed += 1
        if self._total_batches_processed % self.target_update_interval != 0:
            return
        mode = self.target_update_mode
        if mode == "hard":
            self.target_agent_group.load_state_dict(
                {k: v.clone() for k, v in self.eval_agent_group.state_dict().items()}
            )
            self.target_critic.load_state_dict(
                {k: v.clone() for k, v in self.eval_critic.state_dict().items()}
            )
        elif mode in ("ema", "polyak"):
            self._ema_update(
                self.target_agent_group, self.eval_agent_group, self.target_update_tau
            )
            self._ema_update(
                self.target_critic, self.eval_critic, self.target_update_tau
            )
        else:
            raise ValueError(f"Unknown target_update_mode '{mode}'")

    # ------------------------------------------------------------------
    # Parameter synchronisation — all_reduce across workers
    # ------------------------------------------------------------------

    def synchronize_target_params(self):
        """Average ``target_agent_group`` and ``target_critic`` across all
        workers via ``dist.all_reduce``.

        Called at the end of each epoch (after local polyak updates) to
        ensure all workers operate on the same target-parameter baseline,
        preventing error accumulation from per-worker independent polyak
        trajectories.  After this call the master reads the consensus
        state from worker 0.
        """
        for net in (self.target_agent_group, self.target_critic):
            if net is None:
                continue
            for param in net.parameters():
                dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
                param.data /= self.world_size

    def reduce_gradients(self):
        """All-reduce gradients across all workers for the standard
        ``eval_critic`` and ``eval_agent_group`` modules.

        Subclasses that own additional modules (e.g. ``ssl_model`` for
        self-supervised variants, ``eval_v_net`` for QTRAN) override this
        and call ``super().reduce_gradients()`` first, then add the extra
        networks' gradient all-reduce.
        """
        for net in (self.eval_critic, self.eval_agent_group):
            for param in net.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= self.world_size

    # ------------------------------------------------------------------
    # Learning-rate sync (standard critic + agent optimizers)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Master-sync params
    # ------------------------------------------------------------------

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
