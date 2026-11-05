# Group communication: MAPPO, QMIX and relational NP

## Supported path

`GroupCommunicationMAPPO` and `GroupCommunicationQMIX` inherit the shared
`GroupCommunicationAgentGroup`. Each sender encodes only its own observation
window (padding-aware GRU), emits a fixed number of scalar codes, and the group
fuses the decoded messages using two attention stages (16 inducing points,
8 output slots by default). Each agent queries that memory with its own history.
Group IDs must be in `[0,N)` for live agents; dead senders contribute no messages.
Training reuses collected group IDs instead of recomputing communities.

Four component combinations use `use_np: false/true` and `hyper_rank: 0/positive`.
The latter generates a signed low-rank residual action head while retaining the
same attention reader. All agents currently share one observation/action schema
and one model ID. This implementation uses synchronous complete group snapshots;
it does not model wireless delivery, delay, or packet loss.

Set both rollout and replay `required_attrs` to `group_communication_mappo` or
`group_communication_qmix`. The registry owns codes, snapshot steps and group
fields. Cached behavior codes are audit records; gradients use freshly encoded
observations. QMIX independently re-encodes next observations with online and
target networks. MAPPO preserves collected action log probabilities. Existing
profiles and old episode datasets remain supported.

`message_dim: 16/32/64` with `bits: 4` means 64/128/256 payload bits. Replay stores
unpacked uint8 codes, **not** packed network bytes. `codec.pack/unpack` implements
actual payload packing. `MessagePacket` adds a 9-byte versioned envelope; use
`snapshot_packets` to reject mixed snapshots and deduplicate identical sender
retransmissions. Different senders are never deduplicated by feature similarity.
Rollout now serializes each live sender payload and its envelope and records
`communication_bytes`. Quantized payloads are packed codes; raw payloads are
float32 local features; `message_mode: local` emits no messages. This counts one
logical broadcast per sender, not relay transmissions or a packet-level network
simulation. Group control traffic and link-layer headers are additional costs.
For grid position metadata configure `message_map_size`; without it envelope
coordinates are zero placeholders, not inferred global positions.

For image observations, `mask_observation_channels` masks the last channel axis
before both local-policy and message encoding. It requires `(B,N,T,H,W,C)`.
The complementary-pursuit configuration must mask observation channels 3 and 6;
do not apply those indices to its different global-state schema.

## NP and supervision

The prior is conditioned only on group messages. The training posterior also
reads masked query/label pairs; one sampled group latent serves all queries.
Loss is `(sum query BCE + group KL)/number of valid queries`, with outer weight
`auxiliary_weight` (default 0.1). Empty groups/queries contribute zero.
Actions use deterministic prior parameters, never posterior samples or labels.
`np.predict(context, queries, samples)` samples the prior, sharing each draw
across queries to retain their joint dependence. This architecture does not by
itself guarantee calibrated uncertainty; calibration must be measured.

`relation_loss` accepts `relation_queries`, `relation_labels`, `relation_mask`
with shapes `(B,G,Q,D)`, `(B,G,Q)`, `(B,G,Q)` and the recorded group's slot index.
`relation_targets` optionally constructs labels at training time from state.
The provided `StateThresholdQueries` is a generic interface check: fixed state
indices, fixed query descriptors and thresholds. It is **not** a MAgent capture
oracle. `ComplementaryAttackQueries` now provides 14-feature spatial existence,
complementary-hit and capture queries for the 25-channel native state. Query
locations are sampled independently of target presence. Its damage/HP settings
must match the environment; absent attackers and duplicate candidate cells are
handled explicitly. Sparse positive events need monitoring during the pilot;
a passing label test does not establish sufficient supervision density. Privileged state is used here only
for labels and the central critic, never for the actor's forward input.

## Teacher and curriculum

1. Train the matching algorithm/config with `message_mode: raw`. It exchanges
   unquantized local features; its legal observations and groups match the student.
2. For matching architectures, initialize the quantized student's compatible
   parameters from the teacher. For differing components, train the message
   interface first and use the explicit readout handoff below. Create
   `CommunicationCurriculum(student, teacher, stage_steps=(...,...,...))`.
3. Pass recorded observations, padding, alive masks and group IDs to `step`.
   The stages replace one sender, half of each group rounded up, then all senders.
   The frozen teacher fuses the replacement tokens with its raw tokens. Teacher
   parameters receive no gradients, but replacement-token gradients remain live.
   Supply `batch` to add the same relation loss used by online training.
4. Use `readout_step` to train the student's own fusion/NP/head on frozen learned
   messages, especially when its architecture differs from the teacher. Then use
   `save_checkpoint/load_checkpoint` to save/restore model, optimizer, stage,
   readout counter and Torch CPU/accelerator RNG states together. Reconstruct the
   instance from the same experiment config; changing backend is not bitwise
   reproducibility.
   After warm-up, start fresh quantized rollouts; never treat mixed-teacher actions
   as on-policy PPO data.
5. Optionally configure `teacher_checkpoint` (the matching raw actor `.pth`),
   `distillation_steps` and `distillation_weight` for online linear annealing.
   An independent `teacher_config` allows the same teacher checkpoint across
   NP/hypernetwork variants instead of cloning the student's architecture.
   MAPPO distills masked policy KL; QMIX distills available Q values with MSE.
   The update counter and frozen teacher are in the actor state dict. The source
   teacher file must remain available when reconstructing this configured agent.
   Include config and optimizer state in any full training-resume archive;
   the existing `save_current_model` API remains a model-weight checkpoint.

No teacher is automatically trained by loading a checkpoint. No performance or
optimality guarantee follows from these utilities.

## Local smoke configurations

`examples/group_communication_mappo.yaml` and
`examples/group_communication_qmix.yaml` use the small MPE spread environment.
They test data and optimization plumbing, not the complementary-capture method.
Run using the activated project environment:

```sh
python main.py train --config=examples/group_communication_mappo.yaml
python main.py train --config=examples/group_communication_qmix.yaml
```

On this Mac, use the OpenMP runtime setup documented in `c04_environment.md`.
The component suite also exercises MPS when available. CPU worker-loss tests
bypass distributed collectives; they do not validate multi-GPU communication.
No remote training is part of this implementation check.


## Complementary-pursuit CP01 integration

Use `module_name: marlite.environment`, `env_name: complementary_pursuit`.
`n_pursuers`, `n_evaders`, `map_size` and `wall_density` control exact populations;
unsupported capacity or >1024 native IDs is rejected. `prey_policy` selects a
seeded random policy or a fixed local-window escape policy. Native rewards and
attack dynamics remain unchanged. Pursuers alone are exposed to the learner.
Density channels 3/6 are zeroed in the wrapper, before any teacher/student sees
observations. Global state remains separate; wall occupancy is refreshed on reset.

`feature_extractor` accepts the existing ModelConfig: grid inputs remain HWC and
can use Permute/Conv2d/pooling/Flatten/Linear; output one `hidden_dim` vector per
frame. Omit it for the original flatten/Linear path (checkpoint names retained).
The constructor never guesses that every tensor should be processed by a CNN.
Agent feature extractors reject BatchNorm and nonzero Dropout: batch statistics
can couple independent senders, and stochastic training features invalidate the
current deterministic probability-replay contract. Use per-sample normalization
such as LayerNorm/GroupNorm when needed.

The wrapper validates predicted captures against native captures each step.
`training_snapshot` is privileged validation data, never a policy observation.
Use rollout/replay profiles `complementary_mappo`/`complementary_qmix` and analyzer
`complementary` to make terminal capture fraction the first selection metric.
Effective-hit rate counts target-events; missed-attack rate counts individual
attack actions. Neither substitutes for final capture fraction.

`examples/complementary_mappo.yaml` and `examples/complementary_qmix.yaml` are
3-pursuer/5-evader, 15×15 smoke runs. Set `message_mode` to `local`, `raw` or
`quantized` for controlled paths. When changing populations, update agent names,
QMIX input count and group-builder native IDs together: evaders are assigned
IDs first, followed by pursuers. State ID bits are channels 5..14; presence 3 is
pursuers. The example's PartialMAgent radius is 8, refresh interval 4, and group
capacity equals pursuer count (use at least 2 for singleton tests: the legacy
PartialMAgent API treats `n_groups <= 1` as disabling community detection). Group snapshots assume routed group delivery;
they do not claim every pair of group members is one-hop reachable.

`encoder_lr_scale: 0.25` gives features/history/codec one quarter of policy LR;
worker LR synchronization preserves the ratio. If using OneCycleLR, supply
per-group maximum learning rates so that the scheduler does not overwrite it.
PPO hard-code KL early stopping, gradient-angle/code-flip diagnostics and the
formal multi-seed evaluation harness are still separate work before long runs.
Do not treat smoke-test capture rates or native-event consistency as evidence
that communication improves task performance.

Capture labels at the float32 HP reconstruction boundary are masked rather than
asserted as certain. Runtime capture checks accept the corresponding bounded
ambiguity; the reported capture count always comes from native entity removal.
