# Sequence architecture migration

Baseline: `7a54e00`; branch: `codex/trajectory-refactor`.

This is an intentionally incompatible replacement. Historical experiments stay
on the C04 branch. No old YAML, import, or checkpoint compatibility is promised.
The legacy trainers remain runnable only while the new pipeline is being built.

## Acceptance gates

| Gate | Deliverable | Status |
| --- | --- | --- |
| R0 | Migration inventory and mathematical reference tests | Inventory recorded; existing reference tests retained |
| R1 | CPU tensor Episode, T+1 frames, continuous sequence slicing | Implemented; see test/test_sequence_contract.py |
| R2 | Explicit state, step/sequence protocol, GRU and causal attention | Implemented: local composed AgentGroup, window wrapper, GRU and causal reference |
| R3 | Shared multi-timestep MAPPO learner, fixed GAE, independent evaluation | Implemented; local API, CPU and real MPE2 short training validated |
| R4 | torchrun and one explicit gradient reducer | CPU two-rank path validated; server GPU acceptance pending |
| R5 | QMIX, QPLEX, QTRAN sequence learners | Implemented; CPU, Gloo and real MPE2 short workflows validated |
| R6 | Graph, messages, consensus, SSL and gradient mixers | Implemented; CPU and two-rank Gloo verified; GPU acceptance remains R9 |
| R7 | C04 offline/online phases and all paper variants | Staged Python API implemented; CPU/Gloo verified; recipe configuration follows at R8 |
| R8 | New YAML, CLI, analyzer, examples; remove old hierarchy | New configuration/CLI/analyzer verified; legacy cleanup and adapter migration in progress |
| R9 | Multi-environment, resume and performance acceptance | Pending |

## Migration inventory

| Existing trainer registration | New algorithm and composition |
| --- | --- |
| MAPPO | MAPPO learner + local policy |
| GraphMAPPO | MAPPO learner + graph communication |
| QMIX | QMIX learner + local Q head |
| GraphQMIX | QMIX learner + graph communication |
| QPLEX | Independent QPLEX learner, full-action Q and dueling mixer |
| QTRAN | Independent QTRAN learner, counterfactual Q and required V network |
| MsgAggr | QMIX learner + deterministic message aggregation |
| ProbMsgAggr | QMIX learner + probabilistic aggregation |
| GroupConsensusQMIX | QMIX learner + Gaussian consensus |
| SSLGroupConsensusQMIX | Above + AE/VAE reconstruction and KL objectives |
| SSLGroupConsensusMAPPO | MAPPO learner + consensus + SSL objectives |
| VAEGraphQMIX | QMIX learner + graph + VAE objectives |
| DistilledGroupCommunicationMAPPO | Teacher/student AgentGroups + staged C04 recipe |

Reference tests: test_multistep_training, test_mappo_minibatches,
test_value_learning, test_loss_mixer, test_g2anet_policy_consistency,
test_distilled_group_communication. Freeze RNG/noise for stochastic comparisons.
New persistent recurrence is not numerically equivalent to the old zero-start
sliding window. Compare objectives on stateless one-step batches first.

## Data and temporal contract

Episode fields are detached CPU tensors. Slots follow `agent_ids`, never dict
insertion order. Frames have T+1 entries; transitions have T. Producers tensorize
communication data and include any padding masks. All bool masks mean valid.
Sequence batches require matching slot and feature specifications. Do not mix
maps with different agent counts in a fixed-N mixer batch.

`agent_reset` marks a new life before processing the frame. The final frame must
be the actual environment final observation; Episode cannot infer whether a
producer accidentally supplied an auto-reset observation. Environment adapters
must test that invariant. Team boundaries are explicit, not inferred from empty
env.agents. Termination dominates truncation. A timeout bootstraps but stops GAE.

SequenceSlice specifies the first learning transition, learning length, preceding
burn-in, and future target tail. No sequence crosses episodes. Learning targets
cover every transition once when partitioning an episode. SequenceBatch trace
masks stop at the batch boundary; full-episode GAE must be computed before slicing.

Temporal encoders use B,T,N,D, return every step, and accept caller-owned state.
Collectors own state per environment, actor, critic, teacher and student. Clear
state at reset/death/reappearance. Fix policy weights during collection rounds.
No network stores implicit environment state. Attention has bounded per-agent
cache; its step-based reference sequence path is causal, but is not yet a fused
Transformer training implementation. Uninterrupted GRU chunks use one fused call.

Reconstruction defaults for the forthcoming learners: PPO cached_burnin (32
learning steps, 8 warmup); off-policy zero_burnin (32 learning, 32 warmup), with
separate online/target reconstruction. Both are approximations. prefix_recompute
replays the complete prefix under current parameters for each optimizer batch.
Cache identity/version is a learner responsibility; never reuse student state
for teacher or behavior state for target. Warmup runs without autograd.

## Remaining implementation constraints

Trainer owns stages; learners own algorithm models/optimizers. Communication
components do not duplicate PPO or TD. QPLEX needs full action Q; QTRAN's V is
mandatory. SSL retains gate/detach semantics and four mixers. C04 codec injection
must backpropagate through the frozen teacher, unlike teacher reference targets.
Only frozen feature caches can survive an optimizer step.

Use torchrun, one learner/GPU, rank-0 collection/evaluation and immutable shared
CPU episode files (torch.load mmap). One explicit gradient reduction path handles
ordinary and PCGrad updates. Reduce loss sums by global valid counts, preserve
globally-unused grad=None, and project PCGrad only after global task gradients.
No algorithm-specific Worker/WorkerGroup copies, parameter averaging, multi-node,
asynchronous training, sharding, or automatic restarts in this iteration.

New YAML selects recipe/components/constructor parameters; Python owns control
flow and freezing. Reject unknown parameters. Save resolved config/code/dependency
versions. Standard training defaults to independent evaluation, no rollback, no
evaluation replay. Preserve old weight-only rollback/target reanchoring and
validation replay as explicit off-policy options; never add final-test data.

Checkpoints include every optimizer, scheduler, target, mixer, phase/counter,
random stream, sampler and replay/offline-data reference. Save at collection or
offline-update boundaries. Exact resume assumes unchanged device topology.

Final acceptance includes 1-vs-2/3-device gradient parity with uneven masks,
checkpoint continuation, all paper switches, environment adapters, and measured
time/memory. No long research training is part of this refactoring task.

The replacement pipeline is executable through SequenceMAPPOTrainer's Python
API, not yet through the old CLI/YAML. Do not send new configs to the old
TrainerConfig. YAML/CLI replacement is R8. All-rank checkpoints must be called
collectively. Pure on-policy data is removed after every completed distributed
iteration; final inference does not sample a fictitious action.

New tests are independent of old trainer fixtures. test_sequence_mappo covers
collection, fixed labels, exact/cached replay, death/respawn, frozen value updates,
independent evaluation, exact continuation and real MPE2. test_sequence_distributed
runs torchrun with two CPU/Gloo ranks, including empty/uneven partitions, unused
parameters, full checkpoint continuation and single/global-batch comparison.
macOS tests force IPv4 loopback rendezvous to avoid broken IPv6 reverse DNS.
This is not yet GPU validation or a completed migration of the remaining methods.

## Foundation validation (2026-11-19)

73 passed, 1 skipped across test_sequence_contract, test_return_estimation,
test_value_learning, test_loss_mixer, and test_mappo_minibatches on local CPU.
There are 21 new contract tests. The legacy multiprocessing tests require both
repository and parent directory on PYTHONPATH under pytest importlib mode.
No server, GPU, full environment training, or new learner integration was tested.


## Sequence MAPPO validation (2026-11-19)

33 passed across the three new sequence suites, including a real MPE2 environment
and two torchrun CPU configurations. A first torchrun attempt timed out before
launching ranks because the local address resolved to an IPv6 reverse-DNS name;
explicit IPv4 rendezvous resolves it. Tests clean the entire process group on
failure. GPU, off-policy, communication/SSL and C04 migration remain pending.


## Value-learning migration (R5)

`ValueLearner` executes a shared update with separate QMIXObjective,
QPLEXObjective and QTRANObjective. Mixers compose TeamSequenceEncoder with
monotonic mixing, duplex transformation/joint attention, or counterfactual agent
features. QTRAN retains a separately optimized required V network. n-step targets
use real final-frame endpoints; online and target reconstruct their own states.
Hard/EMA refreshes advance only after an actual valid learning update.

Both trainers now inherit one shallow TrainingLifecycle for collection,
evaluation, logging and checkpoints. They contain different iteration policies,
not different GPU math. SequenceValueTrainer adds episode replay, environment-step
based epsilon decay, default held-out validation, explicit optional weight-only
rollback/target reanchoring and opt-in validation replay. Final-test evaluation
has a separate random stream and is never added to replay. Replay and rollback
weights are embedded in full checkpoints so expiring mmap files cannot invalidate
resume. Evicted runtime files are released only after every rank finishes reading.

EpisodeReplay supports uniform or explicitly named reward_weighted chunk sampling.
The latter is neither TD-error PER nor importance-corrected sampling. n-step
returns remain uncorrected off-policy returns, as in the old implementation.

New tests independently calculate n-step sums/endpoints and a QTRAN counterexample,
check QMIX monotonicity, QPLEX decomposition/IGM and detached advantages, illegal
Q masking, counterfactual action independence, target timing, disabled losses,
optimizer ownership, replay eviction, opt-in policies and exact resume. All three
algorithms execute real MPE2 short workflows. torchrun/Gloo tests compare global
updates against single-process updates with uneven and empty ranks, including
QTRAN replay/rollback continuation. Server GPU acceptance remains R9 work.

State checkpoints are read directly for warmup, never duplicated into training
windows or copied wholesale to GPU. Off-policy collection does not save behavior
hidden state. PPO still records per-frame state; sparse attention-cache recording
must be addressed before large-context scalability acceptance at R9.

R5 regression: 149 passed, 1 skipped (local CPU, 2026-11-19), including
three real MPE2 value-learning workflows and real two-rank torchrun/Gloo tests.
The skipped case belongs to existing optional distributed coverage; no server
GPU claim is made. Full migration and old-code removal remain in progress.


## Communication and representation migration (R6)

SequenceAgentGroup composes optional message extraction, independent message
recurrence, communication, local/received joins and action readout. There is no
communication-specific PPO/TD implementation. GraphCommunication handles recorded
availability or differentiable G2ANet weights; dense graph and installed PyG
models are adapters. SetCommunication supports masked mean/max/attention pooling.
GaussianConsensus retains AE, mean of samples, precision product and diagonal
information-proportional CI rules. The CI rule does not guarantee calibration
of arbitrary learned latent beliefs. Dual-path policies carry `message.*` state
alongside local state; prefix reconstruction restores both independently.

Environment-side context providers tensorize fixed/dynamic groups or adjacency
at EVERY frame. Privileged topology construction is identified in episode
metadata. Global state does not automatically enter the actor. Actual learned
adjacency is recorded per transition for graph reconstruction labels. Policy
sampling noise is collected once and replayed, keeping PPO's behavior ratio
consistent. Auxiliary Monte Carlo noise is separate, keyed by update/episode/
frame and independent of rank partitions and overlapping chunk order.

AuxiliaryLearning composes a standalone head objective, optimizer, optional task
mixer and successful-update warmup counter with either learner. Reconstruction
supports deterministic or variational agent/group representations, group or
live-member population weighting, agent/group KL, MSE/BCE and unreduced custom
set losses. Target constructors flatten B*T into singleton histories so their
last-frame interfaces label every learning step. Mask direction and retained
time axes are explicit; environment labels remain training-only. Existing MAgent
semantic features are tested through the adapter. The policy's RL gate and
message/fusion detach switches remain independent from SSL.

Weighted sum, PIT, EMA norm balancing and PCGrad reuse existing numerical cores.
New learners supply local sums/global counts and request SUM gradients. PIT uses
global loss values before its CDF; gradient mixers measure global task gradients
before nonlinear balancing/projection. PCGrad contributes one world-size fraction
of projected shared gradients per rank while private heads use the ordinary
reducer. Zero-weight auxiliary heads retain grad=None and optimizer momentum does
not advance. Full checkpoints contain auxiliary heads/optimizers/mixer buffers
and counters. Logs aggregate each loss by its own valid count, including when SSL
starts partway through an iteration.

During migration the retained Chamfer core was corrected to exclude padded
points BEFORE nearest-neighbour search. A hand-calculated counterexample covers
this and all-empty sets; padding must not artificially lower reconstruction loss.

R6 regression: 222 passed, 1 skipped (local CPU, 2026-11-19), including real MPE2
communication/VAE short training and actual two-rank tests for all four algorithms
with all four task mixers, uneven/empty ranks and exact checkpoint continuation.
The skipped reference case is optional; server GPU tests remain outstanding.
C04 staged training, YAML/CLI replacement and old-code cleanup are not complete.


## C04 staged migration (R7)

GroupSetCommunication composes two-level set fusion with individual attention
queries. QuantizedCommunication adds a local scalar codec without exposing group
IDs or partner features to the sender encoder. AgentGroup.decode_actions is an
explicit readout boundary: frozen local features can be reused while message
replacement reruns the frozen teacher's aggregation/readout with autograd.
The hard codec now uses hard.detach()+(soft-soft.detach()) so its forward values
are exactly the transmitted integers, including rounding boundaries. Tests
require exact equality with decoding the recorded codes, without a threshold.

FrozenPolicyReference independently reconstructs complete teacher histories and
keeps a bounded CPU cache of local features/logits. It never borrows student state
or caches a differentiable aggregation result. The Trainer owns one immutable
teacher; the online objective merely references it. Optional online KL decays by
successful online updates and is independent of the optional query head. Query
heads are standalone Algorithm modules; training labels/queries remain outside
the actor. Per-frame CPU sampling makes queries and progressive sender replacement
invariant to batch order and distributed partitions.

SequenceDistillationTrainer adds shallow phase control to SequenceMAPPOTrainer:
optional teacher-context probe fitting, codec fitting through the frozen teacher,
student policy fitting with local encoder/codec frozen, then ordinary MAPPO.
The two main offline stages share the same recorded teacher episodes. Validation
episodes use a separate seed stream. All/one/half/all sender replacement schedules,
optional stage-specific auxiliary weights, skipped offline training and optional
online distillation are explicit switches. Offline query weights override the
online query coefficient rather than multiplying it implicitly.

Checkpoints can be saved after each offline minibatch, not only online iterations.
They embed teacher/critic weights, training/validation episodes, phase/counters,
probe, all phase optimizers, sampler RNG and validation measurements. Derived
teacher caches are cleared on restore. Online actor/critic/auxiliary state remains
owned by the shared learner. Completed probes remain frozen after restoration.
No algorithm-specific distributed worker is introduced.

Received-code interventions can replace all senders or only selected senders,
keeping local history queries unchanged. Collection records the actual codes and
configured message-byte budget. Payload pack/unpack remains exact; configured
header bytes describe the simulation protocol, not physical radio traffic.

R7 regression: 244 passed, 1 skipped (local CPU, 2026-11-19). New coverage includes
complete real-MPE2 short C04 training, independent module freezing, optional
components, before/after offline validation, per-phase and online exact resume,
and complete two-rank Gloo workflows with uneven/empty online partitions. Single
and distributed results compare identifiable parameters, optimizer states and
actual policy outputs. Attention's shared key-projection bias is a softmax-null
direction: tiny FP32 residual gradients can yield different Adam drift there;
only that functionally irrelevant parameter slice is excluded from numerical
weight equality, with its invariance separately tested. Same-topology resume
remains bitwise tested. Server GPU acceptance, YAML/CLI migration, experiment
recipes and removal of legacy code/tests remain outstanding.


## Configuration and entry-point migration (R8, first acceptance)

The version-3 configuration selects five Python recipes and constructor-injected
components. One lazy registry covers neural modules, communication, topology,
labels, objectives and optimizers. Unknown types/arguments and injected-field
replacement fail with paths. Bounded dimension references replace shape guesses;
there is no eval or YAML training/loss/stage graph. Explicit initializers and
strict checkpoint boundaries retain the original initialization semantics.

All thirteen former trainer families have runnable composition examples under
examples/sequence, with additional dual-path, G2ANet and pooled-attention cases.
Sixteen MPE2 examples execute short training and actor-only evaluation workflows.
A seventeenth example configures the native four-pursuer/eight-evader capture task,
CNN, state-based groups and task-specific optional relation queries. This example
still requires server-native acceptance; it is not counted as locally verified.

The installed `marlite`, `python -m marlite`, and source-checkout main.py share
train/validate/analyze commands. C04 may train its teacher with the same standard
MAPPO recipe before transferring an independent frozen copy. Teacher fitting never
runs during construction/validation. Offline phase optimizers, probe architecture,
policy parameter groups, neural components and the RL gate are configurable.
Invalid online iteration counts fail before potentially costly teacher fitting.

Policy-only evaluation builds no critic, replay, teacher or training head. It loads
actor weights independently of training world size and uses a held-out random
stream. Exports are actual CPU tensor Episodes. Native environment metrics, action
counts, graph edge counts and configured communication budgets have tensor-based
summaries. Value-free episodes cannot accidentally enter PPO target preparation.
Native metric extraction happens before closing the simulator. Simulator probes
and collection belong only to rank 0, not one expensive simulator per learner.

Runs save resolved YAML, dependency/Git provenance, file hashes and a source
archive, including dirty source files. Full checkpoints validate configuration;
only run limits/workdir may change on exact resume. Final checkpoints are saved
regardless of periodic-checkpoint settings. A fresh run refuses an occupied run
directory. The package entry point was reinstalled locally with --no-deps, leaving
the existing Torch/environment packages unchanged.

Topology constructors run in evaluation mode so configured update intervals work.
Their Python/NumPy/CPU-Torch random streams are isolated and episode-seeded. A null
experiment seed still shares ONE sampler stream between ranks. Implicit random
activations/dropout and batch-dependent normalization are rejected consistently
in actors, critics and auxiliary heads. NearestDecoder now uses GroupNorm instead
of BatchNorm so padding and rank partitioning cannot change valid examples.

R8 first acceptance: 292 passed, 1 skipped in the migrated regression matrix
(local CPU, 2026-11-19). This includes installed CLI discovery, actual CLI
train/resume/analyze, all five recipes under two-rank Gloo, independent teacher
training, source/version artifacts and all sixteen short MPE2 example workflows.
New configuration modules pass the formatter and E9/F checks. This is not yet an
all-repository cleanup acceptance: old APIs/tests/examples still exist.

A local native-MAgent2 attempt aborted with libomp duplicate-runtime initialization
(exit 134) before its workflow could be verified. No KMP_DUPLICATE_LIB_OK workaround
was enabled; its temporary output was removed. Native/GPU acceptance is moved to
the authorized Linux server, whose inspected environment has Torch
2.15.0+rocm7.14. Only read-only server inspection has occurred at this stage; no
training queue or long experiment was started.

### Remaining retirement dependencies

Before deleting the old hierarchy, migrate remaining environment opponent
controllers, heterogeneous observation/action padding and per-model routing;
inspect the old AgentGroup variants rather than assuming shared-policy tests
cover them. Preserve relevant initializer, environment and mathematical reference
coverage while rewriting old API tests. Primitive package initializers still
import old configuration factories and must be made independent. Retire old
trainers/workers/WorkerGroups, datasets/replay/rollout/config processors, analyzer
wrappers, superseded model wrappers, YAML examples and associated tests only after
these dependencies have replacements. Pooled/masked model adapters need coverage
for their mask/output conventions. Sparse recurrent-state snapshots, server GPU
parity, native environments and measured sequence performance remain R9 work.
