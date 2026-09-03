"""Scalar schedulers with explicit warmup and exact endpoint semantics."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from numbers import Real


class Scheduler(ABC):
    """Base class for a scalar value that changes over a bounded interval.

    The value equals ``start_value`` through ``ramp_start_step`` and equals
    ``end_value`` from ``ramp_start_step + ramp_steps`` onward.
    """

    def __init__(
        self,
        start_value: float,
        end_value: float,
        ramp_start_step: int,
        ramp_steps: int,
    ) -> None:
        if not all(isinstance(value, Real) for value in (start_value, end_value)):
            raise TypeError("start_value and end_value must be real numbers")
        if not all(math.isfinite(float(value)) for value in (start_value, end_value)):
            raise ValueError("start_value and end_value must be finite")
        if not isinstance(ramp_start_step, int) or isinstance(ramp_start_step, bool):
            raise TypeError("ramp_start_step must be an integer")
        if not isinstance(ramp_steps, int) or isinstance(ramp_steps, bool):
            raise TypeError("ramp_steps must be an integer")
        if ramp_start_step < 0:
            raise ValueError("ramp_start_step must be non-negative")
        if ramp_steps <= 0:
            raise ValueError("ramp_steps must be positive")

        self.start_value = float(start_value)
        self.end_value = float(end_value)
        self.ramp_start_step = ramp_start_step
        self.ramp_steps = ramp_steps

    def get_value(self, step: int) -> float:
        """Return the scheduled value for a non-negative integer step."""
        if not isinstance(step, int) or isinstance(step, bool):
            raise TypeError("step must be an integer")
        if step < 0:
            raise ValueError("step must be non-negative")

        progress = min(
            max((step - self.ramp_start_step) / self.ramp_steps, 0.0), 1.0
        )
        return self.start_value + (
            self.end_value - self.start_value
        ) * self._transform_progress(progress)

    @abstractmethod
    def _transform_progress(self, progress: float) -> float:
        """Map normalized progress in ``[0, 1]`` to a curve position."""


class LinearScheduler(Scheduler):
    """A scheduler that changes linearly after its initial plateau."""

    def _transform_progress(self, progress: float) -> float:
        return progress


class LogarithmicScheduler(Scheduler):
    """A delayed log-space curve with a configurable late-growth rate.

    ``curve_rate=0`` is exactly linear.  Positive rates keep early values near
    the start value and concentrate the change near the end of the ramp.
    """

    def __init__(self, *args, curve_rate: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not isinstance(curve_rate, Real) or not math.isfinite(float(curve_rate)):
            raise TypeError("curve_rate must be a finite real number")
        if curve_rate < 0:
            raise ValueError("curve_rate must be non-negative")
        if curve_rate > 100:
            raise ValueError("curve_rate must not exceed 100")
        self.curve_rate = float(curve_rate)

    def _transform_progress(self, progress: float) -> float:
        if self.curve_rate == 0:
            return progress
        return math.expm1(self.curve_rate * progress) / math.expm1(
            self.curve_rate
        )


class FixedScheduler(Scheduler):
    """A scheduler that always returns one fixed finite real value."""

    def __init__(self, value: float) -> None:
        super().__init__(value, value, ramp_start_step=0, ramp_steps=1)

    def _transform_progress(self, progress: float) -> float:
        return 0.0
