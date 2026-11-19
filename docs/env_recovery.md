# Environment recovery

`ParallelCollector` discards interrupted episodes and retries from a fresh
simulator with the same episode seed. Completed episodes are retained. Both
training and evaluation use the same explicit policy:

```yaml
collector:
  max_steps: 200
  retry:
    max_retries: 3
    initial_delay: 1.0
    max_delay: 30.0
    backoff_factor: 2.0
    jitter: 0.25
```

These are defaults. `max_retries: 0` disables recovery. The budget is per episode,
including constructor/reset/step failures; no interrupted tensor Episode reaches
learning. Delays grow exponentially up to the cap. Jitter uses OS randomness,
without consuming experiment random streams. The failed simulator closes before
waiting, and policy device/training mode is restored even on exhaustion.

Only `EnvironmentInterrupted`, `ConnectionError`, `TimeoutError` and backend
`recoverable_errors` are retried. Invalid actions/configuration and programming
errors propagate immediately. Backend adapters should translate their known
transient errors into `EnvironmentInterrupted`, or declare exception classes in
`recoverable_errors`. SMACWrapper supplies SC2 protocol errors. Its internal
`reset_retries` defaults to zero; avoid stacking two recovery budgets.

A native process abort/segmentation fault cannot be handled by a Python exception
policy. Recovery is not a remedy for native memory corruption.
