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
