"""Environment failures for which discarding and recollecting a rollout is safe."""


class EnvironmentInterrupted(RuntimeError):
    """An environment backend failed; the current episode must be discarded."""


def recoverable_errors(env):
    """Backends may declare extra transient exception types on their wrapper.

    Do not retry arbitrary Exception/OSError: invalid actions, configuration
    errors and programming bugs must remain visible rather than be retried.
    """
    return (EnvironmentInterrupted, ConnectionError, TimeoutError) + tuple(
        getattr(env, "recoverable_errors", ())
    )


class RetryPolicy:
    """Bounded per-episode recovery with jitter isolated from training RNGs."""

    def __init__(
        self, max_retries=3, initial_delay=1.0, max_delay=30.0, backoff_factor=2.0, jitter=0.25
    ):
        import math

        if type(max_retries) is not int or max_retries < 0:
            raise ValueError("max_retries must be nonnegative")
        values = (initial_delay, max_delay, backoff_factor, jitter)
        if any(not math.isfinite(v) for v in values):
            raise ValueError("Retry parameters must be finite")
        if not 0 <= initial_delay <= max_delay or backoff_factor < 1 or not 0 <= jitter <= 1:
            raise ValueError("Invalid retry delay/backoff/jitter")
        self.max_retries, self.initial_delay = max_retries, initial_delay
        self.max_delay, self.backoff_factor, self.jitter = max_delay, backoff_factor, jitter

    def wait(self, attempt):
        import random
        import time

        # Iterative capping avoids exponent overflow for an unusually high budget.
        base = self.initial_delay
        for _ in range(attempt):
            base = min(self.max_delay, base * self.backoff_factor)
        delay = random.SystemRandom().uniform(
            base * (1 - self.jitter), min(self.max_delay, base * (1 + self.jitter))
        )
        if delay:
            time.sleep(delay)
