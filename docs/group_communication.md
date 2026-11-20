# Quantized Messaging via Policy Distillation (QMPD)

The formal method name follows the manuscript's English title, *Fixed-Length
Quantized Message Exchange Between Agents via Policy Distillation*. In MARLite,
QMPD names the quantized communication method and `qmpd` selects its training
recipe. Older run configurations and full checkpoints contain their original
recipe identifier. Use the corresponding historical checkout to resume them
exactly.

Use the `qmpd` recipe with either `examples/sequence/qmpd_mpe2.yaml` or the native
fork environment in `examples/sequence/qmpd_magent2.yaml`. See
[configuration](sequence_configuration.md) for CLI, component and optimizer syntax.
These are interface examples, not paper hyperparameters.

## Stages and ownership

| Stage | Data | Updated | Frozen |
| --- | --- | --- | --- |
| Optional raw teacher training | Fresh teacher trajectories | Teacher actor/critic | Student |
| Optional query-probe fitting | Fixed teacher training episodes | Probe head | Teacher, student |
| Codec fitting | The same fixed teacher episodes | Student codec | Teacher, probe, local encoder and student policy |
| Policy fitting | The same fixed teacher episodes | Aggregation, local readout/action head, optional student query head | Teacher, local encoder and codec |
| Online MAPPO | Fresh student trajectories | Student actor/critic, optional query head | Teacher and probe |

`teacher.source: train` uses a normal SequenceMAPPOTrainer and transfers its
trained policy into a separate frozen reference before any student fitting.
`teacher.source: checkpoint` requires a strict `teacher.actor.checkpoint` spec.
Both sources honor `offline.initialize_from_teacher`. The default copies compatible
teacher modules before student fitting. For online-only training, set
`offline.offline_enabled: false` and `offline.initialize_from_teacher: false` to keep
the student's own initial parameters while retaining the teacher as a KL target.
Teacher and student AgentGroups have independent parameters and recurrent state.

The sender encoder receives only local history. Raw group fusion can be replaced
with QuantizedCommunication, whose scalar codec uses exact hard integers in the
forward pass and straight-through derivatives. GroupSetCommunication aggregates
current group tokens with two learned query sets; each receiver queries group
slots using its local history. The slots are not an additional temporal memory.

Codec training mixes decoded student sends into the frozen teacher. Its local
features/logits can be cached, but its aggregation and action readout must run
with autograd so the message receives a gradient. Policy fitting freezes the
local encoder/codec and distils the teacher's masked action distribution.

## Optional components

`offline` controls episode counts, codec/policy update counts, replacement schedule,
probe updates and `codec_aux_weight`/`policy_aux_weight`. Replacement is `all`, or
`progressive` with three positive `replacement_steps` summing to codec updates:
one sender, approximately half, all. Sampling depends on update/episode/frame,
so changing the number of learners does not change the selected senders.

`auxiliary.objective` selects a standalone RelationQueryObjective/QueryPredictionHead
and training-only label provider. An independent frozen probe is copied from that
spec by default, or supplied as `offline.probe_objective`. Auxiliary signals never
enter the actor. Stage weights override the online query coefficient. For native
capture queries the label generator implements the actual complementary-attack
rules; StateThresholdQueries is a generic smoke-test provider only.

`teacher.online_distillation: {enabled: true, weight: 0.1, decay_updates: 1000}` adds
an optional teacher KL loss during ordinary MAPPO. It is independent of offline
training and auxiliary prediction; `enabled: false` removes it. The successful
online-update counter controls decay, without advancing during offline stages.

`offline.optimizers.codec/policy/probe` configure phase-specific optimizers. Policy
parameter groups use prefixes `aggregation`, `readout`, `action_head`, `auxiliary`.
Online optimizers are configured separately. Phase momentum is intentionally not
reused as online momentum.

## Checkpoints and diagnostics

`run_offline(max_updates=..., checkpoint_interval=...)` permits interactive phase
inspection. Offline checkpoints contain teacher/critic, both datasets, probe,
phase counters, optimizers, sampler RNG and validation measurements. Derived caches
are rebuilt. Same-topology restoration is tested at offline and online boundaries.

Received integer codes can be supplied in execution context as `received_codes`;
`replace_senders` selects which sends to replace. This changes received information
while preserving local history queries. Actual codes and the configured byte
budget are recorded per transition. Header bytes remain a simulation convention,
not a physical-radio traffic or latency measurement.
