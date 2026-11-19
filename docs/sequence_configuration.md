# Sequence configuration, version 3

YAML selects a recipe and component constructor arguments. Python owns stage
order, model/optimizer ownership, graph connections and loss definitions. There
are five recipes: `mappo`, `qmix`, `qplex`, `qtran`, `c04`. Communication variants
are compositions of those recipes rather than additional trainer names.

## Component specifications

```yaml
version: 3
recipe: mappo
seed: 42
environment:
  module: mpe2.simple_spread_v3
  params: {N: 3, max_cycles: 25, continuous_actions: false}
defaults: {hidden_dim: 32, mixing_dim: 16, heads: 4, learning_rate: 0.0005}
collector: {max_steps: 25}
learner: {burn_in: 8, state_strategy: cached_burnin}
trainer: {sequence_length: 16, batch_sequences: 16, ppo_epochs: 4}
run: {iterations: 100, train_episodes: 8, eval_episodes: 16, checkpoint_interval: 10}
```

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
| DistilledGroupCommunicationMAPPO | c04_mpe2.yaml, c04_magent2.yaml |
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
