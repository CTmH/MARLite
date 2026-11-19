# MARLite

A research-oriented MARL library with explicit tensor trajectories, recurrent state
and composed communication modules. The sequence API replaces the old trainer
hierarchy; historical C04 experiments remain on their original Git branch.
Old YAML, imports and checkpoints are intentionally incompatible.

## Run an experiment

Use the project environment with the appropriate PyTorch extra: `cpu`, `mps`,
`cuda`, or `rocm`. For example, `uv sync --extra mps` on Apple Silicon. Environment
packages follow the sources pinned in `pyproject.toml`; the cooperative capture
example requires the CTmH/MAgent2 fork.

```sh
python -m marlite validate --config examples/sequence/mappo.yaml --device cpu
python -m marlite train --config examples/sequence/mappo.yaml --workdir results/mappo --device cpu
python -m marlite train --config examples/sequence/mappo.yaml --workdir results/mappo --resume results/mappo/checkpoints/final.pt --iterations 100
python -m marlite analyze --config examples/sequence/mappo.yaml --checkpoint results/mappo/checkpoints/final.pt --episodes 32 --test-seed 997 --output results/mappo/test.json
```

`--iterations` means additional iterations. A fresh run refuses an already used
work directory. Analysis loads only actor weights, uses a separate evaluation
stream, and can export actual tensor episodes with `--trajectories episodes.pt`.

For CUDA or ROCm, use one identical learner per local GPU:

```sh
OMP_NUM_THREADS=2 torchrun --standalone --nproc-per-node=3 -m marlite train --config examples/sequence/qplex.yaml --workdir results/qplex --device cuda
```

Rank 0 owns simulator construction, collection, evaluation and logging. Learners
share immutable CPU episode files and sum gradients, rather than averaging model
parameters. Only single-node execution is supported. CPU/Gloo uses the same path;
on machines with broken IPv6 resolution use an explicit IPv4 rendezvous endpoint.

## Research interfaces

- Episodes store CPU tensors: T transitions and T+1 real frames. All boolean
  validity masks use True for valid. Terminations, timeouts and new lives differ.
- Encoders expose `step` and `forward_sequence`, accepting explicit recurrent
  state and returning every learning timestep. GRU chunks use the fused path.
- MAPPO, QMIX, QPLEX and QTRAN share execution/lifecycle utilities. Communication
  and auxiliary objectives are injected components, not extra PPO/TD subclasses.
- The `c04` recipe optionally trains a raw-message teacher, then runs codec and
  student-policy offline distillation followed by MAPPO. Online distillation and
  auxiliary prediction are independently optional.
- Checkpoints include optimizers, targets, auxiliary mixers, recurrent replay
  settings, RNG streams and replay/offline data. C04 also saves phase boundaries.

See [configuration and recipes](docs/sequence_configuration.md),
[C04 training](docs/group_communication.md), and the detailed
[migration/acceptance record](docs/trajectory_refactor.md).

The examples demonstrate composition; they are not tuned paper configurations.
Run the rewritten sequence tests with one pytest process when validating actual
multi-process execution. GPU/environment acceptance and legacy-code cleanup are
tracked separately in the migration record.
