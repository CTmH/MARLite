# Initialization and reproducible experiments

Both features are opt-in. Existing configs retain native layer initialization.
Nothing automatically changes MAPPO, QMIX, or SSL weights when loading a config
without `initialization`. Existing server experiments are not modified.

## Experiment seed

```yaml
trainer:
  # Add these to the existing trainer mapping.
  seed: 42
  deterministic: false
```

`seed` is a non-negative integer or null (default). `deterministic` is a boolean
(default false). Strict deterministic mode enables PyTorch's deterministic
algorithms and raises on unsupported operations instead of silently relaxing the
requirement. It may cost performance. It cannot guarantee equality across
hardware, PyTorch versions, or environment implementations.

Stable SeedSequence streams separate online-model construction, early SSL/value
network construction, main-process training, worker ranks, and rollout rounds.
Workers still receive synchronized model parameters from the trainer; distinct
worker RNGs control stochastic training, not independent initial models.

Each rollout task receives its own seed, passed to Python, NumPy, PyTorch and
`env.reset(seed=...)`. Persistent workers reseed per episode. Seed assignment
does not depend on which process finishes first; seeded persistent results are
returned in task order. Both managers support repeated generation without
reusing episode seeds. Evaluation and collection have separate round counters.
KMeans generators derive their per-sample seeds from the seeded NumPy stream.

An environment must actually honor `reset(seed=...)`; accepting a keyword does
not prove that its simulator is deterministic. Seeded reset/collection failures
are surfaced rather than silently returning partial data. No SC2 reproducibility
claim is made. Changing batch size, world size, or sampling settings can still
change training. Checkpoints do not yet save RNG states or rollout counters for
bitwise-exact interrupted-run continuation.

## Model initialization

`initialization` belongs beside `model_type` in any ModelConfig. It is not a
Trainer-wide forced policy. The same config is accepted by the G2ANet graph
builder; its module paths are relative to `attention_model`.

Supported schemes and options:

| Scheme | Options (PyTorch defaults when omitted) |
| --- | --- |
| `orthogonal` | `gain` |
| `xavier_uniform`, `xavier_normal` | `gain` |
| `kaiming_uniform`, `kaiming_normal` | `a`, `mode`, `nonlinearity` |
| `normal` | `mean`, `std` |
| `uniform` | `a`, `b` |
| `trunc_normal` | `mean`, `std`, `a`, `b` |
| `constant` | `value` (required) |

Bias is independent: `bias: 0.0` fills supported biases with zero; omitting it
preserves their native values. Do not select ReLU Kaiming settings automatically
for GELU. Orthogonal/Xavier/Kaiming require at least two dimensions. Use
`constant` for zeros/ones instead of additional scheme aliases.

Rules apply in this order:

1. Construct the model using its native initialization.
2. Apply the model's default rule to supported projections.
3. Apply exact module overrides, including optional output-row slices.
4. Load this model's checkpoint, if configured (on CPU).

Explicitly initialized or pretrained child ModelConfigs are protected boundaries:
parent defaults skip them, and parent overrides targeting them raise an error.
An enclosing model's own checkpoint intentionally supplies its entire state.
Shared parameters are initialized once per default pass. Conflicting overrides
of the same/shared parameters raise; output slices within one override must be
non-overlapping. Module names are exact `named_modules()` paths, not patterns;
`module: ""` selects the root. Invalid targets/options fail instead of being
silently ignored. Lazy parameters must be materialized first.

### MAPPO action output head

This complete decoder block initializes only the final action-logit projection:

```yaml
decoder:
  model_type: Custom
  layers:
    - {type: Linear, in_features: 64, out_features: 64}
    - {type: GELU}
    - {type: Linear, in_features: 64, out_features: 5}
  initialization:
    overrides:
      - module: model.2
        scheme: orthogonal
        gain: 0.01
        bias: 0.0
```

Adapt dimensions and the exact module path to the actual experiment. This is
an experimental starting point, not a guaranteed improvement. Do not apply
the small action-logit gain to every hidden layer or to QMIX mixing weights.

### VAE consensus mean and log variance

For a final `Linear(64, 128)` emitting `[mu(64), log_var(64)]`, use:

```yaml
initialization:
  overrides:
    - module: model.2
      scheme: normal
      mean: 0.0
      std: 0.01
      bias: 0.0
      output_slices:
        - start: 64
          stop: 128
          scheme: normal
          mean: 0.0
          std: 0.001
          bias: 0.0
```

Output slices are half-open and supported only on Linear output rows, including
PyG Linear. They are not inferred from model names. AE heads need no variance
slice. Near-zero individual log variance means variance near one, not negligible
noise; group fusion can change it substantially.

### Recurrent modules

```yaml
initialization:
  overrides:
    - module: rnn
      scheme: xavier_uniform
      bias: 0.0
      recurrent: {scheme: orthogonal, gain: 1.0}
```

GRU/LSTM weights are initialized gate by gate, for all layers and directions.
`input` and `recurrent` are optional complete weight rules, not bias rules.
The outer scheme handles unspecified weights. LSTM additionally accepts
`forget_bias`; it sets bias_ih's forget block and zeros bias_hh's corresponding
block so their effective sum equals the requested value. This option requires
an LSTM with biases. For G2ANet, select `module: hard_attention`.

### Attention, graphs, normalization and other layers

- MultiheadAttention initializes each Q/K/V projection separately, plus the
  output projection exactly once, for packed and separate projection layouts.
- GCN/GAT projection weights include PyG Linear layers. GAT attention vectors
  retain native initialization unless explicitly selected as direct parameters.
- LayerNorm/BatchNorm and positional encodings retain native initialization.
  Direct parameter overrides can change them deliberately; running statistics
  and other buffers are never treated as weights.
- Linear and Conv1d/2d/3d, including transpose variants, use PyTorch tensor-layout
  semantics. ConvTranspose output-row slicing is deliberately unsupported.
- Hypernetworks and mixers initialize their generating networks, not weights
  produced dynamically during forward. Avoid zeroing abs-constrained heads.
- Unsupported Custom-layer types keep native initialization under a default
  rule. Explicit unsupported module overrides fail; a direct tensor override
  remains available when its layout is understood.

Example of an explicit GAT vector rule:

```yaml
initialization:
  overrides:
    - module: conv1
      parameter: att_src
      scheme: normal
      std: 0.01
```

## Optional initial-output diagnostics

`marlite.util.initialization_diagnostics.inspect_initial_outputs` accepts the
same forward arguments as the AgentGroup, plus current-step `action_mask` and
`active_mask`. Call it once on a real collated rollout window, before training
or compilation, and log the returned dictionary with the existing logger.

It evaluates a copy in eval mode and preserves RNG states, so neither dropout,
model/group-builder caches nor diagnostic sampling alters training. It reports
normalized legal-action entropy (excluding forced actions) and the mean/std of
available agent/group mu, log variance and sampled consensus. Dead agents and
empty groups are excluded when their masks/indices are supplied. The one-off
copy costs model memory; this helper is not installed as a per-batch hook.

Suggested comparison: retain identical experiment seeds, first change only the
MAPPO action head, then add the VAE output-head initialization. Do not change
every backbone initializer simultaneously, or attribution becomes difficult.
