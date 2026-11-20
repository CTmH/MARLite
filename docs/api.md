# Python API reference

These interfaces cover the common extension points. For an end-to-end example,
see [Add a custom component](custom_components.md).

## Experiment construction

::: marlite.configuration.recipe.build_experiment

::: marlite.configuration.registry.Registry
    options:
      members: [register, names, build]

## Trajectory and training batches

::: marlite.data.episode.Episode
    options:
      members: [validate, state_dict, from_state_dict]

::: marlite.data.sequence.SequenceSlice

::: marlite.data.sequence.SequenceBatch

## Temporal encoder

::: marlite.algorithm.temporal.TemporalEncoder
    options:
      members: [initial_state, step, forward_sequence]

::: marlite.algorithm.temporal.TemporalOutput
