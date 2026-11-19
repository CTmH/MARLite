# Multi-step TD, GAE and sequence replay

Temporal context, training sequence length, warmup length and reward horizon are
independent. Encoders return a feature at **every** timestep. A longer TD horizon
does not require a longer materialized observation history.

## Value learning

```yaml
recipe: qplex
learner: {gamma: 0.99, n_steps: 3, burn_in: 32, state_strategy: zero_burnin}
trainer: {sequence_length: 32, batch_sequences: 16, updates_per_iteration: 4}
```

For k available transitions, the target is
`sum(gamma**i * reward[t+i]) + gamma**k * Q_target(t+k)`. The horizon shortens at
an episode boundary. Online utilities select legal endpoint actions; target
utilities and mixers evaluate them. Online/target temporal states are reconstructed
independently. No stale behavior hidden state is used for a target network.
Targets are recomputed during updates. This is uncorrected n-step TD, without
importance correction or TD(lambda); larger horizons can increase off-policy bias.
Team rewards use explicit `sum` or `mean` aggregation.

## MAPPO

```yaml
recipe: mappo
learner: {burn_in: 8, state_strategy: cached_burnin}
trainer:
  gamma: 0.99
  gae_lambda: 0.95
  normalize_advantages: true
  sequence_length: 32
  ppo_epochs: 4
collector: {state_snapshot_interval: 32}
```

GAE uses complete collected episodes and frozen behavior values, including the
real final frame. Set lambda to zero for one-step residuals. Labels stay fixed
for all PPO epochs; normalization affects advantages, not value targets.
`cached_burnin` begins at the nearest saved behavior snapshot preceding the
requested warmup. It therefore may replay up to one extra snapshot interval.
`zero_burnin` discards prior context; `prefix_recompute` replays from frame zero
under current weights before every update. The first two strategies approximate
current-policy recurrent state after parameters change; exact prefix replay costs
more compute. Sparse state tensors stay on CPU, outside minibatch frames.

## Boundaries

True termination stops bootstrap and the advantage trace. A time/collection limit
stops the trace but allows bootstrap from actual surviving-agent observations.
Termination takes precedence when both flags are set. Individual death does not
end the team's return. New lives reset their own state; unborn slots remain dead.
An auto-reset observation must never replace the final frame. Missing real final
data raises instead of silently reusing stale observations.

See `test_sequence_contract.py`, `test_sequence_mappo.py`, `test_sequence_value.py`
and `test_util/test_return_estimation.py` for mathematical and integration checks.
