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
The envelope is a standalone transport utility, not a packet-level rollout
simulator. Group control traffic and link-layer headers are additional costs.

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
oracle. Native AA/AB/AAB event constructors and their environment validation are
still required before real C04 experiments. Privileged state is used here only
for labels and the central critic, never for the actor's forward input.

## Teacher and curriculum

1. Train the matching algorithm/config with `message_mode: raw`. It exchanges
   unquantized local features; its legal observations and groups match the student.
2. Initialize a matching quantized student from that trained teacher's actor
   state (same architecture/budget), then create
   `CommunicationCurriculum(student, teacher, stage_steps=(...,...,...))`.
3. Pass recorded observations, padding, alive masks and group IDs to `step`.
   The stages replace one sender, half of each group rounded up, then all senders.
   The frozen teacher fuses the replacement tokens with its raw tokens. Teacher
   parameters receive no gradients, but replacement-token gradients remain live.
   Supply `batch` to add the same relation loss used by online training.
4. Save both `curriculum.state_dict()` and `optimizer.state_dict()`; reload both
   before resuming. The curriculum includes the teacher weights and stage counter.
   After warm-up, start fresh quantized rollouts; never treat mixed-teacher actions
   as on-policy PPO data.
5. Optionally configure `teacher_checkpoint` (the matching raw actor `.pth`),
   `distillation_steps` and `distillation_weight` for online linear annealing.
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
