# Group communication: MAPPO, QMIX and relational NP

> **Experimental (C04).** Group communication for MAPPO, QMIX and QPLEX,
> relational NP, the optional conditional head, the message curriculum and the
> complementary-pursuit benchmarks are research interfaces. Configuration keys
> remain available for reproducing earlier runs; these experiments do not
> change the status of the stock algorithms.

## Implementation

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
   The default stages replace one sender, half of each group rounded up, then
   all senders. Set `replacement="all"` for the direct-distillation control;
   it uses the same update budget and readout handoff.
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

For experimental single-device quantized MAPPO, set
`trainer.experimental_message_target_kl` to a positive threshold (for example,
`0.02`). A fixed minibatch first checks the collected hard codes and action log
probabilities against the pre-update policy. After each PPO pass the trainer
recomputes hard codes and measures the masked categorical KL against that
policy. Exceeding the threshold skips the remaining passes; it does not roll
back the last pass. If the initial training-device forward disagrees with
rollout, the audit replays the same actor on the rollout device one team
window at a time. Hard codes must match that replay exactly. The existing
action log-probability check still applies. On mismatch, a reproduction
bundle under `message_audit/` records the sample IDs, observations, masks,
behavior weights, and outputs of both paths. An unreproducible rollout stops
training; a verified rollout records the finite backend discrepancy separately
from the pre/post-update KL. No quantization tolerance is used, and current
policy forwards never consume cached rollout codes. This option is disabled by default and is not a guarantee
on the rest of the rollout or on individual code changes.

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

## Experimental CP04 run preparation

A matching CP03 raw checkpoint can be converted into a paired teacher and
initialization bundle without importing temporary CP03 training code:

```sh
python -m marlite.experimental.cp04 prepare \
  --source-config /path/to/cp03/config.yaml \
  --source-actor /path/to/cp03/checkpoints/final/agent/agent.pth \
  --seed 42 --output /path/to/new/bundle
```

Preparation strictly checks the source actor's weights, initializes a fresh
codec and critic, and records SHA-256 hashes. The manifest explicitly records
that real trajectory parity has not yet been verified. Retain the bundle as
the common initialization across paired variants.

Use a YAML file containing separate nonempty lists under
`offline_train`, `offline_validation`, `online_validation`, and `test`:

```sh
python -m marlite.experimental.cp04 collect \
  --bundle /path/to/bundle --split-file /path/to/splits.yaml \
  --output /path/to/new/offline_data
```

Only the two offline lists are passed to the raw teacher. Episode files store
numeric tensors and retain sender order; `load_offline_dataset` reconstructs
the original padded trajectory windows and checks that the data belong to the
same teacher. Online training uses a separate seed stream, which the online
runner must record and audit against these held-out splits.

Fit the relation head once from the shared teacher/data bundle, then pass the
same probe to both warm-up variants:

```sh
python -m marlite.experimental.cp04 fit-relations \
  --bundle /path/to/bundle --data /path/to/offline_data --output /path/to/probe
python -m marlite.experimental.cp04 warmup \
  --bundle /path/to/bundle --data /path/to/offline_data --probe /path/to/probe \
  --replacement progressive --output /path/to/new/warmup
```

Use `--replacement all` for the matched full-replacement control. The probe
reports per-query label coverage and Brier scores against the training prior.
The joint warm-up rejects a probe without positive and negative support in
both splits for every query type, or without pooled Brier improvement. For
a short wiring test, `--kl-only` replaces `--probe`; its output is explicitly
marked KL-only. It is not an evaluation of the full supervised method.

Both variants consume the same deterministically selected windows and
queries. Codec training freezes the student history/front-end and teacher;
the readout phase additionally freezes the learned codec and trains the
student fusion and heads. `actor.pth` contains only the student. The separate
`offline_state.pth` can be resumed with `--resume` and the same arguments;
it preserves both optimizer states and validates input manifests and settings.

The CP04 probe now uses a `group_geometry` proposal. It samples candidate
cells in attack range of the group's recorded live sender positions, favoring
jointly reachable cells for half the queries. It never reads target existence
or HP to choose a query; those fields supply labels only. Legacy query configs
still use the uniform proposal unless explicitly changed.

Start a new single-device run and evaluate a selected checkpoint separately:

```sh
python -m marlite.experimental.cp04 online \
  --bundle /path/to/bundle --data /path/to/offline_data --probe /path/to/probe \
  --offline /path/to/warmup --rounds 10 --distillation-steps 100 \
  --device cpu --output /path/to/new/online
python -m marlite.experimental.cp04 evaluate \
  --run-directory /path/to/new/online --data /path/to/offline_data \
  --checkpoint /path/to/new/online/checkpoints/final/agent/agent.pth \
  --split online_validation --output /path/to/validation.yaml
```

The initial student weights and online teacher are independent. Omit
`--offline` to start from the bundle's student initialization; supply it to
reuse an offline handoff. A positive `--distillation-steps` enables the
bundle's frozen raw teacher. With zero steps, no online teacher is loaded.
Omitting both gives the ordinary quantization control. The runner starts
with empty replay, checks that the derived rollout
seed schedules do not collide with reserved splits, and records configuration,
hashes and schedules. Validation/test episodes never enter replay. Test mode
requires a previously selected checkpoint SHA-256 and is disabled for
`--kl-only-smoke` runs; that option allows one round without a relation probe
for wiring verification. MAPPO evaluation samples actions; QPLEX uses greedy
execution. Fixed rounds are not an exact environment-step budget.

For a formal relation-loss ablation, use `warmup --kl-only` to omit offline
relation supervision and `online --no-online-relations` to omit online
relation gradients. Supply the eligible probe to the online command even
when disabling gradients: the label proposals still consume the same
random draws. Neither option disables action distillation or changes the
model architecture. The separate `--kl-only-smoke` flag still limits an
unprobed run to one round and cannot be used for formal test evaluation.

Add `--structured` without `--offline` or teacher distillation to run the
experimental structured control. Its fixed local encoder sends four
4-byte records `[x, y, HP, type]` (nearest visible prey first, then visible
walls, then padding). It uses the same 16-byte payload, 9-byte envelope,
history for the local policy, and trainable decoder/fusion/readout. The
fixed encoder has no trainable parameters; report that difference when
comparing model sizes. The records are derived from the sender's current
legal view, with no global state or other senders as encoder input.

### Experimental paper controls and frozen references

Set `online --paper-baseline masia|ss_gigc|coalign` to use an experimental
128-bit adaptation with its own auxiliary loss. Do not supply a CP04 relation
probe or teacher-distillation budget for these controls. All use the same
local observation schema, grouping, 16-byte payload, packet envelope,
actor initialization for shared tensors, and fresh critic initialization.

| Control | Preserved mechanism | Adaptation |
|---|---|---|
| `masia` | Permutation-invariant common representation, local extraction, reconstruction and future prediction | Shared attention context, per-agent gate, current/one-step next 4×4 occupancy targets; forecast conditioned on a group action histogram |
| `ss_gigc` | Diagonal Gaussian precision-weighted group fusion, mean used for actions, reconstruction with prior KL | Four bits each for 16 means and 16 log variances; reconstructs the coarsened union of positives in the group's legal views |
| `coalign` | Attention aggregation and collaborative-to-global Gaussian KL | Global belief learned by an occupancy VAE during training, rather than the manuscript's critic-integrated global value representation; the collaborative mean feeds the actor |

These are matched-budget mechanism controls, not reproductions of the
original architectures or reported performance. The global state and future
state appear only in the training losses. The policy forward is deterministic
given the hard messages; latent sampling is confined to auxiliary objectives.
The run manifest includes total and requires-grad parameter counts. A
requires-grad count can include shared components unused by a given control.

Use `reference --bundle ... --data ... --split online_validation --output ...`
to evaluate the raw teacher on the same maps as the quantized variants. Supply
both `--local-config` and `--local-actor` for a compatible frozen local-only
reference. The local architecture, environment and history must match.
Archived references remain historical checkpoints, not equal-training-budget
controls. Test mode requires the locked reference weight hash.

### Experimental budget and diagnostics

For matched training interactions, add `online --environment-steps N`. Set
`--rounds` high enough to reach the cap even when episodes end early. The
persistent manager shortens the last collection batches through native
episode limits, preserving the truncation/bootstrap fields. Interrupted
episode retries are disabled in exact-budget mode. If the round limit ends
first, the manifest stays incomplete and records the observed count.
Fixed-round mode remains available for small diagnostics.

The online manifest separates successful collection steps, trainer-only
diagnostic steps and independent validation/test; records actual rollout
seeds, episode lengths, optimizer step counts and online wall time. Offline
artifacts report data generation, probe and warm-up costs separately. The
warm-up compares the initial codec, warmed codec and trained readout on the
same fixed offline validation windows. The codec metric uses the unchanged
teacher readout with all tokens replaced; the student metric uses its own
readout. MAPPO records hard-message KL, clip fraction and code flips.
When relation supervision is available, both trainers log the first
minibatch's RL/relation gradient cosine on the codec, excluding distillation
from the RL term; a zero or unusable norm is not reported as zero cosine.

Use `evaluate --greedy` or `reference --greedy` for supplemental MAPPO
evaluation; it respects available actions and never changes training rollout.
