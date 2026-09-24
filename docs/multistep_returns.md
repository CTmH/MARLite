# Multi-step TD and GAE

History length (`rollout.traj_len` and `replay_buffer.traj_len`) and return
horizon are independent. Keep the two history lengths equal. The policy and
critic still emit one prediction per input window; no recurrent state or
sequence-to-sequence interface is required.

## Off-policy value learning

```yaml
trainer:
  type: QMIX
  gamma: 0.99
  n_steps: 3
```

`n_steps` is a positive integer, defaulting to `1`. All OffPolicyTrainer
variants and their workers consume the same discounted reward sum and
bootstrap discount. For k available transitions, the target is
`sum(gamma**i * reward[t+i]) + gamma**k * Q_target(t+k)`.
Actions at the endpoint are selected by the online agent and evaluated by
the target agent/mixer, as before. Targets are evaluated during each update,
not permanently cached in replay. Current observations, rewards, and SSL
targets remain unchanged; only the next window moves to the n-step endpoint.
The configured team reward aggregation (`sum` or `mean`) is preserved.

This is **n-step TD**, not TD(lambda), Q(lambda), or an importance-corrected
off-policy return. Larger horizons can increase off-policy bias.

## MAPPO

```yaml
trainer:
  type: MAPPO
  gamma: 0.99
  advantage_estimator: gae
  gae_lambda: 0.95
  normalize_advantages: false
```

These options also apply to GraphMAPPO and SSLGroupConsensusMAPPO.
`advantage_estimator` defaults to `one_step`; `gae` enables GAE.
`gae_lambda` is in [0, 1], with zero equivalent to a one-step residual.
Optional normalization affects only policy advantages, not value targets.

Before any PPO/SSL updates, the trainer evaluates the selected episodes'
windows in bounded batches with the critic in eval mode and without gradients.
It computes team advantages backwards along actual transitions, then attaches
scalar advantages and value targets to sampled positions. The labels and the
sampled positions stay fixed for all PPO epochs, including multi-GPU workers.
Even `one_step` now freezes labels before updates instead of recomputing them
from a changing critic inside each minibatch.

Sequence critics receive full history windows for both current and next
values. Stateless critics/mixers continue to select the last state internally.
Two-dimensional per-agent states and their time-aligned alive masks are kept
intact. GPU memory for target inference is bounded by the training batch size.

## Boundaries and agent survival

- A true team termination stops both reward accumulation and bootstrap.
- An artificial time/collection limit stops the return/GAE trace but permits
  bootstrap from the **actual final** observation/state and surviving agents.
- If a time limit is part of the task's terminal objective, the environment
  must report termination, not merely truncation.
- When both flags are true, termination wins. Individual agent deaths do not
  stop the team's trace. Policy losses still exclude agents dead at the
  current decision. Rewards on a death transition remain part of the return.
- Empty `env.agents` at a timeout does not imply physical death: surviving
  agents are recovered from termination flags, and their final observations
  must be available. Missing final data causes an error rather than a stale
  state being silently used for bootstrap. Auto-reset wrappers must expose
  the terminal transition, not substitute the next episode's reset state.
- Short episodes (including a single transition) are accepted. Near reset,
  the next window includes the reset observation and independently updated
  padding. At the end, the horizon is shortened without crossing episodes.
- Replay collected by older code can contain stale final states or all-dead
  timeout masks. Recollect data; these errors cannot reliably be repaired
  from an old buffer.

Target-generation tests live in `test/test_util/test_return_estimation.py`,
rollout boundary tests in `test/test_rollout/test_return_boundaries.py`, and
real-model/worker checks in `test/test_trainer/test_multistep_training.py`.
