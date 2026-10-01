# Environment rollout recovery

Both `persistent-env` and `multi-process` rollout managers accept this optional
configuration under the existing top-level `rollout` section. It applies to
training and evaluation. Omitted fields use the defaults shown below.

```yaml
rollout:
  # Keep the other rollout settings here.
  env_retry:
    max_retries: 3
    strategy: exponential_jitter
    initial_delay: 30.0
    backoff_factor: 2.0
    max_delay: 120.0
    jitter: 0.3333333333333333
```

- `max_retries`: additional recovery attempts, excluding the initial attempt.
  The budget is shared by all episodes in one worker task, as in the original
  recovery implementation; successful episodes do not replenish it.
- `strategy`: `fixed` (constant interval), `exponential` (exponential backoff),
  `fixed_jitter` (randomized constant interval), or `exponential_jitter`
  (randomized exponential backoff, the default).
- `initial_delay`: first retry's base delay, in seconds. Zero disables waiting.
- `backoff_factor`: multiplier for each subsequent retry, at least 1.
  Used only by exponential strategies.
- `max_delay`: upper bound on both base and actual wait; must be at least
  `initial_delay`.
- `jitter`: relative random variation, between 0 and 1. Zero disables jitter.
  Used only by strategies ending in `_jitter`.
  The wait is uniform in `[base * (1 - jitter), min(max_delay, base * (1 + jitter))]`.

Defaults give waits of 20–40, 40–80, then 80–120 seconds. `max_retries: 0`
disables recovery entirely. Invalid settings fail when the manager is built.

All environments can use this policy for `reset()` or `step()` failures:

- `ConnectionError`, `TimeoutError`, and MARLite's `EnvironmentInterrupted`
  are recoverable by default.
- Backends/wrappers can declare a `recoverable_errors` tuple containing extra
  transient exception classes. SMACWrapper declares SC2 protocol errors here.
- Other exceptions (for example `ValueError` for an invalid action) fail
  immediately; they are not silently retried.

For a custom backend, translate its known transient failures at the boundary:

```python
from marlite.environment import EnvironmentInterrupted

# Inside the environment's reset()/step() implementation:
try:
    return backend.step(actions)
except BackendDisconnected as error:
    raise EnvironmentInterrupted("Backend disconnected") from error
```

The old environment is closed before waiting and recreating it.
Normal resets do not wait. Environment-constructor retry behavior is unchanged.
The SMAC wrapper now defaults to `reset_retries: 0` so it forwards failures to
rollout without immediate internal restarts. Remove any explicit nonzero
`environment.wrapper.reset_retries` setting (or set it to 0) to avoid stacking
the wrapper's legacy immediate retries with this policy.
The interrupted episode is discarded and recollected using the same seed;
completed episodes are retained on successful recovery. OS-backed randomness
keeps jitter separate from experiment RNG streams, including across processes.
Exhaustion still raises an error rather than returning an interrupted trajectory.

This is crash recovery, not a fix for native SC2 memory corruption.
