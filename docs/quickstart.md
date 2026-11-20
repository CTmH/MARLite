# Quickstart

Train a recurrent MAPPO policy for three agents in MPE2 Simple Spread. The example
runs two short training iterations and evaluates the policy after each iteration.
Use commands from the root of the MARLite checkout.

## Set up the environment

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and use
Python 3.13. Choose the PyTorch extra for your platform.

```sh
# Apple Silicon
uv sync --python 3.13 --extra mps

# Linux with a CPU-only PyTorch installation
uv sync --python 3.13 --extra cpu
```

The commands use `uv run --no-sync` to reuse the selected environment. The
quickstart explicitly runs on the CPU, so the same experiment works with the
Apple Silicon wheel. Training with `--device mps` requires an available MPS
device in the current session.

## Validate and train

```sh
uv run --no-sync marlite validate \
  --config examples/quickstart/mappo.yaml --device cpu

uv run --no-sync marlite train \
  --config examples/quickstart/mappo.yaml --device cpu \
  --workdir results/quickstart
```

Validation prints the resolved environment and actor specification. Training
records metrics in `results/quickstart/events.jsonl`, resolved settings in
`resolved.yaml`, and run provenance in `provenance.json`. The final policy is
included in `checkpoints/final.pt`. Use a fresh work directory for a new run.

## Evaluate the saved policy

```sh
uv run --no-sync marlite analyze \
  --config examples/quickstart/mappo.yaml --device cpu \
  --checkpoint results/quickstart/checkpoints/final.pt \
  --episodes 2 --test-seed 2026 \
  --output results/quickstart/evaluation.json
```

The evaluation report summarizes rewards and episode lengths. An optional
`--trajectories` path saves the evaluated episodes for later inspection.
To continue training in the same work directory, supply the saved checkpoint
with `train --resume` and use `--iterations` to set the number of additional
training iterations.

## Example configuration

```yaml
--8<-- "examples/quickstart/mappo.yaml"
```

See the [YAML configuration reference](sequence_configuration.md) to change
the environment, sequence length, architecture, or training algorithm.
