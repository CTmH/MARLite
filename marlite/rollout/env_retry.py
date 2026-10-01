"""Bounded Environment recovery backoff, independent of experiment randomness."""

from dataclasses import dataclass
import logging
import math
import random
import time


@dataclass(frozen=True)
class EnvRetryPolicy:
    """Retry budget per rollout worker task (not per successful episode).

    Only interrupted environment episodes use this policy, not normal resets.
    Strategy is fixed/exponential, optionally suffixed with _jitter.
    Set initial_delay=0 for immediate retries or max_retries=0
    to disable recovery. All delays are in seconds.
    """

    max_retries: int = 3
    strategy: str = "exponential_jitter"
    initial_delay: float = 30.0
    backoff_factor: float = 2.0
    max_delay: float = 120.0
    jitter: float = 1 / 3

    def __post_init__(self):
        if self.strategy not in ("fixed", "exponential", "fixed_jitter", "exponential_jitter"):
            raise ValueError(
                "env_retry.strategy must be fixed, exponential, fixed_jitter, or exponential_jitter"
            )
        if type(self.max_retries) is not int or self.max_retries < 0:
            raise ValueError("env_retry.max_retries must be a non-negative integer")
        for name in ("initial_delay", "backoff_factor", "max_delay", "jitter"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"env_retry.{name} must be a finite number")
        if not 0 <= self.initial_delay <= self.max_delay:
            raise ValueError("env_retry requires 0 <= initial_delay <= max_delay")
        if self.backoff_factor < 1:
            raise ValueError("env_retry.backoff_factor must be >= 1")
        if not 0 <= self.jitter <= 1:
            raise ValueError("env_retry.jitter must be between 0 and 1")

    def wait(self, attempt: int, error: Exception):
        """Called after closing the failed environment; attempt is zero-based."""
        if attempt >= self.max_retries:
            raise RuntimeError("Environment episode failed after recovery attempts") from error
        base = self.initial_delay
        if self.strategy.startswith("exponential"):
            for _ in range(attempt):
                base = min(self.max_delay, base * self.backoff_factor)
        # OS entropy avoids changing policy sampling and episode seed streams.
        delay = base
        if self.strategy.endswith("_jitter"):
            delay = random.SystemRandom().uniform(
                base * (1 - self.jitter),
                min(self.max_delay, base * (1 + self.jitter)),
            )
        logging.warning(
            "Environment recovery %d/%d: waiting %.2fs before recreating environment; error: %s",
            attempt + 1, self.max_retries, delay, error,
        )
        time.sleep(delay)
