# Initialization and reproducible experiments

Top-level `seed` is a nonnegative integer or null; `deterministic` defaults to
false. Configure randomness before building models. Strict mode raises for
unsupported deterministic operations, and does not promise equality across
hardware or library versions.

Models are synchronized once at construction in torchrun. Episode seeds derive
from the experiment seed, stream, round and episode index. Training collection,
validation, final testing and C04 offline data use separate streams. Checkpoints
save sampling RNGs, round counters and per-rank random state. Exact continuation
requires the same device topology/configuration. A null experiment seed still
shares one sampler order across ranks. Simulator seed support must be tested;
accepting `reset(seed=...)` alone is insufficient.

## Explicit component initialization

Every registry neural specification may contain `initialization` alongside
`type`/`params`, followed optionally by a strict `checkpoint`:

```yaml
action_head:
  type: Sequential
  params:
    layers:
    - {type: Linear, params: {in_features: 64, out_features: 64}}
    - {type: GELU}
    - {type: Linear, params: {in_features: 64, out_features: 5}}
  initialization:
    overrides:
    - {module: "2", scheme: orthogonal, gain: 0.01, bias: 0.0}
```

Supported schemes: orthogonal, Xavier uniform/normal, Kaiming uniform/normal,
normal, uniform, truncated normal and constant. Their explicit PyTorch options
are validated. Omitting initialization preserves native weights and RNG use.
Bias options are independent. Norm running statistics are not overwritten.

Construction order is native initialization, parent rule, exact overrides, then
checkpoint loading. Explicitly initialized/checkpointed child components are
protected from an enclosing default initializer. Exact module/parameter paths,
nonoverlapping output slices, gatewise recurrent initialization and LSTM forget
biases are supported. Shared parameters are initialized once; conflicting rules
raise. Graph-attention parameters use their actual named module paths.

`inspect_initial_outputs` accepts the policy's ordinary step/sequence arguments
and `PolicyOutput`. It copies the model/inputs, preserves RNGs, ignores padded
agents/groups and reports normalized legal-action entropy and latent statistics.
Use once for diagnostics rather than every minibatch.
