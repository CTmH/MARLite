"""Configuration factory for scalar schedulers."""

from __future__ import annotations

from marlite.util.scheduler import (
    FixedScheduler,
    LinearScheduler,
    LogarithmicScheduler,
    Scheduler,
)


registered_schedulers = {
    "linear": LinearScheduler,
    "logarithmic": LogarithmicScheduler,
    "fixed": FixedScheduler,
}


class SchedulerConfig:
    """Build a registered scalar scheduler from a configuration mapping."""

    def __init__(self, **kwargs) -> None:
        self.scheduler_type = kwargs.pop("type")
        self.scheduler_kwargs = kwargs

    def get_scheduler(self) -> Scheduler:
        try:
            scheduler_class = registered_schedulers[self.scheduler_type]
        except KeyError as error:
            raise ValueError(
                f"Unsupported scheduler type: {self.scheduler_type}"
            ) from error
        return scheduler_class(**self.scheduler_kwargs)
