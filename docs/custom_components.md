# Add a custom component

This example replaces the frame encoder in a recurrent MAPPO policy with a
linear projection followed by GELU and LayerNorm. The GRU still handles temporal
state. MAPPO still owns rollouts, optimization, and checkpointing.

A frame encoder sees a batch of individual observations. MARLite folds the
batch/time/agent axes into that leading axis before calling it. Return a tensor
with one feature vector per observation and set `temporal.params.input_dim` to
the encoder's output width. LayerNorm allows each row to be normalized
independently of the sampled batch or rank partition.

## Define and register the component

Register the class before building the experiment. The class name used by YAML
is exactly the name you pass to `registry.register()`. The launcher passes that
registry to `build_experiment`, so a new experiment script can extend the set of
components independently of the defaults.

```python
--8<-- "examples/extensions/custom_frame_encoder.py"
```

## Select it in YAML

The environment provides `observation_size` and `action_dim`. The encoder, GRU,
and action head each use a hidden width of 16. The action head's output width
comes from the environment, so the launcher does not have to encode it in Python.

```yaml
--8<-- "examples/extensions/custom_frame_encoder.yaml"
```

## Run the example

From the root of the checkout, after completing the [quickstart setup](quickstart.md),

```sh
uv run --no-sync python examples/extensions/custom_frame_encoder.py \
  --workdir results/custom-frame-encoder
```

The example writes the same metrics and final checkpoint as the quickstart.
Use this launcher for the example. Its registry contains `NormalizedFrameEncoder`.
For a custom temporal component, implement [TemporalEncoder](api.md#temporal-encoder)
and have the caller pass its state explicitly.
