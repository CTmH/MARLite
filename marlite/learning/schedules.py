"""Learning-rate schedules driven by explicit, checkpointed training events."""

from dataclasses import dataclass
import math

from torch.optim.lr_scheduler import ReduceLROnPlateau


@dataclass
class ScheduleBinding:
    """A scheduler belongs to one optimizer and one event stream.

    Update events occur only after that optimizer steps. Validation events may
    use a held-out metric, but never standalone/final-test evaluation. A plateau
    scheduler requires an explicit metric; ordinary schedulers must omit it.
    """

    scheduler: object
    event: str = "update"
    metric: str | None = None

    def __post_init__(self):
        if self.event not in ("update", "validation"):
            raise ValueError("Scheduler event must be update or validation")
        needs_metric = isinstance(self.scheduler, ReduceLROnPlateau)
        if needs_metric != (self.metric is not None):
            raise ValueError("Only plateau schedulers require an explicit metric name")
        if self.metric is not None and (not isinstance(self.metric, str) or not self.metric):
            raise ValueError("Scheduler metric must be a nonempty name")


class OptimizerSchedules:
    """Small composition shared by learners and offline stages, without workers.

    Validation schedules advance only if their optimizer has stepped since the
    previous validation. Counters are independent for each optimizer, including
    optional heads. Metrics passed here must already be reduced across ranks.
    """

    def __init__(self, bindings=None):
        self.bindings = dict(bindings or {})
        self.updates = dict.fromkeys(self.bindings, 0)
        self.validated = dict.fromkeys(self.bindings, 0)

    @staticmethod
    def _step(binding, metrics):
        if binding.metric is None:
            binding.scheduler.step()
            return
        if binding.metric not in metrics:
            raise ValueError(f"Missing scheduler metric {binding.metric!r}")
        value = float(metrics[binding.metric])
        if not math.isfinite(value):
            raise ValueError("Scheduler metric must be finite")
        binding.scheduler.step(value)

    def after_update(self, role, metrics):
        if role not in self.bindings:
            return
        binding = self.bindings[role]
        self.updates[role] += 1
        if binding.event == "update":
            self._step(binding, metrics)

    def after_validation(self, metrics, *, roles=None):
        selected = self.bindings if roles is None else roles
        for role in selected:
            binding = self.bindings.get(role)
            if binding is None or binding.event != "validation":
                continue
            if self.updates[role] == self.validated[role]:
                continue
            self._step(binding, metrics)
            self.validated[role] = self.updates[role]

    def state_dict(self):
        return {
            role: {
                "type": type(binding.scheduler).__qualname__,
                "event": binding.event,
                "metric": binding.metric,
                "updates": self.updates[role],
                "validated": self.validated[role],
                "scheduler": binding.scheduler.state_dict(),
            }
            for role, binding in self.bindings.items()
        }

    def load_state_dict(self, state):
        if state.keys() != self.bindings.keys():
            raise ValueError("Checkpoint scheduler roles differ")
        for role, binding in self.bindings.items():
            saved = state[role]
            if saved["type"] != type(binding.scheduler).__qualname__:
                raise ValueError("Checkpoint scheduler types differ")
            if (saved["event"], saved["metric"]) != (binding.event, binding.metric):
                raise ValueError("Checkpoint scheduler event/metric differs")
            binding.scheduler.load_state_dict(saved["scheduler"])
            self.updates[role], self.validated[role] = saved["updates"], saved["validated"]
