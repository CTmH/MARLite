# Sequence configuration, version 3

YAML selects a recipe and component constructor arguments. Python owns stage
order, model/optimizer ownership, graph connections and loss definitions. There
are five recipes: `mappo`, `qmix`, `qplex`, `qtran`, `qmpd`. Communication variants
are compositions of those recipes rather than additional trainer names.

## Configuration keys

The [quickstart](quickstart.md#example-configuration) is a complete version-3
configuration. The following keys control which components run and how they
are trained.

| Key | Role |
| --- | --- |
| `version`, `recipe` | Required. Set `version: 3` and choose one of the five recipes. MAPPO and QMPD use logits and a centralized value model. QMIX, QPLEX, and QTRAN use individual action values, a joint value model, and replay. |
| `environment` | Required. Name the Python module exposing `parallel_env` and pass constructor `params`. Add `wrappers` in application order. |
| `evaluation_environment` | Optional alternate environment with the same trained slot, observation, state, and action specification during training. |
| `seed`, `deterministic` | Defaults are `42` and `false`. A null seed leaves the experiment unseeded. |
| `defaults`, `dimensions` | Set fallbacks for built-in models or define positive integer dimensions for custom constructors. |
| `actor`, `critic`, `objective` | Optional component specifications. `objective` selects a value-learning objective and is only valid for off-policy recipes. |
| `communication_context` | Optional provider of per-frame group or graph data for a communication-enabled actor. QMPD defaults to `AllTogether`. |
| `collector`, `learner`, `trainer` | Configure the environment rollout, optimizer updates, and training loop, respectively. |
| `replay` | Off-policy only. `capacity` counts complete episodes. `sampling` is `uniform` or `reward_weighted`. |
| `optimizers`, `schedulers`, `auxiliary` | Configure optimization and optional auxiliary learning. |
| `teacher`, `offline` | QMPD only. See the [QMPD guide](group_communication.md) for the teacher sources and stage switches. |
| `run` | Supply `iterations` for training. Set `train_episodes`, `eval_episodes`, and the evaluation and checkpoint intervals. |

## Time and batch settings

| Setting | Meaning |
| --- | --- |
| `collector.max_steps` | Cap on environment steps per episode. Default: 200. |
| `collector.device` | Collection device. Default: `cpu`. The CLI's `--device` controls the learner. |
| `collector.state_snapshot_interval` | Interval for caching recurrent state on the CPU. Default: 32 environment steps. |
| `trainer.sequence_length` | Number of learning transitions in each full chunk. Default: 32. |
| `trainer.batch_sequences` | Maximum number of chunks in a training batch across all learners. Default: 32. |
| `trainer.ppo_epochs` | MAPPO and QMPD. Passes over the collected on-policy chunks. Default: 4. |
| `trainer.updates_per_iteration` | QMIX, QPLEX, and QTRAN. Replay updates after collection. Default: 4. |
| `learner.burn_in`, `learner.state_strategy` | Choose how to reconstruct recurrent state before the learning window. |
| `learner.n_steps` | Off-policy return horizon. Default: 1. |

For on-policy learning, configure `gamma` and `gae_lambda` under `trainer`.
For value learning, configure `gamma` under `learner`. See [multistep
returns](multistep_returns.md) for truncation, bootstrapping and state replay.
`run.eval_interval: null` disables periodic validation and
`run.checkpoint_interval: null` disables periodic snapshots. A completed
run with a work directory still saves a final checkpoint.

## Component specifications

Any actor/critic/objective can be replaced with a `type` and `params` specification.
Nested specs construct ordinary Python modules/objects; lists and plain mappings
remain constructor data. `Sequential` takes a `layers` list. Unknown types/fields
and omitted constructor arguments fail with their configuration path. YAML
anchors are allowed; duplicate explicit keys are rejected.

```yaml
actor:
  type: SequenceAgentGroup
  params:
    frame_encoder:
      type: Linear
      params: {in_features: {$dim: observation_size}, out_features: 32}
    temporal:
      type: RecurrentGRU
      params: {input_dim: 32, hidden_dim: 32}
    action_head:
      type: Linear
      params: {in_features: 32, out_features: {$dim: action_dim}}
```

Available environment dimensions are `n_agents`, `observation_size`, `state_size`,
`observation_shape`, `state_shape`, `action_dim`, `joint_action_dim`. A reference
may select `index` from a shape and apply integer `scale`/`offset`. For example,
`{$dim: n_agents, scale: 16}` gives a QMIX weight-generator output size. Custom
positive integer dimensions can be declared under `dimensions`. No expression
execution, object sharing or loss/stage control language is provided.

`initialization` uses the explicit rules in `util.initialization`. A child
initialization/checkpoint is an ownership boundary that outer initialization may
not overwrite. `checkpoint: {path: ..., key: [learner, actor]}` loads a strict
state dictionary. Missing or incompatible weights fail; there is no random-weight
fallback. Paths are interpreted from the invocation directory; prefer absolute
paths for archived experiments.

The registry is inspectable through `default_registry().names`. Extensions are
registered in Python, e.g. `registry.register("MyEncoder", MyEncoder)`, then passed
to `build_experiment(..., registry=registry)`. There are no YAML import paths for
network classes. Environment modules expose their normal `parallel_env` factory;
optional `environment.wrappers` use registered constructor specs with `env`
injected by the environment factory. Only rank 0 probes/starts simulators.

## Optimizers, auxiliary objectives and recurrent replay

`optimizers.actor/critic/objective` contain normal optimizer specs. Parameter
ownership is supplied by the recipe. Optional optimizer `groups` select named
parameter prefixes and set per-group options; remaining parameters form the
last group. Overlapping or unmatched prefixes fail.

```yaml
optimizers:
  actor:
    type: AdamW
    params: {lr: 0.0005}
    groups:
    - prefixes: [frame_encoder, temporal]
      params: {lr: 0.000125}
```

`auxiliary` contains `objective`, optional `optimizer`, `mixer` and `start_update`.
Heads own their parameters independently of the actor/critic. Weighted sum, PIT,
EMA gradient-norm balancing and PCGrad use globally normalized label populations.
Warmup counts successful learner updates, not old epochs. The actor's RL gate and
message/fusion detach flags are separate controls. BatchNorm and unrecorded
stochastic layers are rejected: they invalidate replay or partition-independent
losses. Use LayerNorm/GroupNorm and explicit latent randomness.

On-policy recurrence supports `cached_burnin`, `zero_burnin`, `prefix_recompute`.
Value learners allow only independent `zero_burnin` or `prefix_recompute` for
online and target networks. `replay` belongs only to value recipes. Rollback and
validation replay are explicit trainer options, both disabled by default; final
test data is never inserted into replay.

## Migration examples

| Former trainer/family | New example under examples/sequence |
| --- | --- |
| MAPPO | mappo.yaml |
| GraphMAPPO | graph_mappo.yaml, g2anet_mappo.yaml |
| QMIX | qmix.yaml |
| GraphQMIX | graph_qmix.yaml |
| QPLEX | qplex.yaml |
| QTRAN | qtran.yaml |
| MsgAggr | message_qmix.yaml |
| ProbMsgAggr | probabilistic_message_qmix.yaml |
| GroupConsensusQMIX | group_consensus_qmix.yaml |
| SSLGroupConsensusQMIX | ssl_group_consensus_qmix.yaml |
| SSLGroupConsensusMAPPO | ssl_group_consensus_mappo.yaml |
| VAEGraphQMIX | vae_graph_qmix.yaml |
| DistilledGroupCommunicationMAPPO | qmpd_mpe2.yaml, qmpd_magent2.yaml |
| Dual-path encoders | dual_path_message_qmix.yaml |
| Pooled attention history | attention_history_mappo.yaml |

The pooled attention example explicitly wraps its history model with
`WindowEncoder`; it is a sliding-window reference, not a fused Transformer
sequence trainer. `RecurrentGRU` and `CausalAttention` use the native explicit-state
contract. Gaussian fusion rules and target definitions are configurable separately.
A mean-observation reconstruction target is a generic baseline, not entity
association/deduplication.

## Reproducibility and analysis

Runs save `resolved.yaml`, `provenance.json` and a source archive with file hashes,
Git state and dependency versions. Checkpoints validate the mathematical config;
run limits/workdir may change on resume. Exact continuation requires the same
number of learners/devices. Source snapshots record dirty code as well as commits.

`analyze` loads only the actor, independent of training device count. It reports
rewards, lengths, per-agent action counts, communication budgets, graph edge counts
and native environment metrics when available. Exports contain CPU tensor Episodes,
not legacy dictionaries. The critic is not constructed during policy evaluation,
which permits changes to central-state shape when the actor's schema still fits.

## Learning-rate schedules and heterogeneous agents

Schedulers are optional, owned by each optimizer role. A binding selects an
`update` event (the default) or an explicit held-out `validation` event:

```yaml
schedulers:
  actor:
    scheduler: {type: StepLR, params: {step_size: 1000, gamma: 0.5}}
  critic:
    scheduler: {type: ReduceLROnPlateau, params: {mode: max, patience: 10}}
    event: validation
    metric: reward_mean
```

Update schedules advance after that optimizer actually steps. Plateau schedules
require a named, globally reduced metric; other schedulers omit `metric`.
Validation schedules do not advance without intervening optimizer updates.
Standalone evaluation and final testing never alter learning rates. All scheduler
states/counters are checkpointed. `teacher.schedulers` and `offline.schedulers`
use the same schema; offline roles are `probe`, `codec`, `policy`, with validation
at stage completion. Top-level roles are actor/critic/objective/auxiliary; an
absent optimizer cannot have a scheduler.

Heterogeneous environments can explicitly wrap `PadAgentView`. It pads flattened
local observations (optionally appending feature validity) and discrete actions.
For private or role-shared networks, compose `RoutedFrames`, `RoutedTemporal`
and `RoutedHead` with one stable role per agent slot. Each role owns its modules
and recurrent state. No loss, optimizer or trainer subclass is needed. Roles
without valid samples retain `grad=None`, including AdamW momentum/decay behavior.
Flat feature-prefix lengths are explicit; convolutional roles should receive an
appropriate structured observation adapter instead of silently reshaping padding.

`GaussianParameters` adapts tuple-returning encoders. `MaskedPool` specifies the
wrapped model's mask keyword, True convention and output convention explicitly.
For example, `ResAttMaskedProbEnc` accepts `alive_mask=True` and returns one
concatenated tensor, not a tuple. `WindowEncoder(use_padding_mask: false)` adapts
plain CNN/sequential history models; these remain finite-window encoders.

For an end-to-end example of choosing a Python component from YAML, see
[Add a custom component](custom_components.md).
