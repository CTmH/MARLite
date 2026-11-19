# Complementary pursuit environment

`examples/sequence/c04_magent2.yaml` uses complementary pursuit from the pinned
CTmH/MAgent2 fork, including native A/B attack rules and optional training-only
relation-query labels. Use the appropriate `cpu`, `mps`, `cuda` or `rocm` extra.

```sh
python -m marlite validate --config examples/sequence/c04_magent2.yaml --device cpu
python -m marlite train --config examples/sequence/c04_magent2.yaml --workdir results/c04 --device cuda
MARLITE_NATIVE_MAGENT2=1 python -m pytest test/test_native_magent2.py
```

Native tests are opt-in before import, so ordinary CPU tests do not accidentally
launch simulators or load incompatible native libraries. Actual native acceptance
must run separately; a skip is not evidence of integration correctness.

Apple Silicon installations may load separate OpenMP runtimes from Torch and
MAgent2. Use a consistently linked native build, or perform native acceptance on
the Linux server. Do not enable `KMP_DUPLICATE_LIB_OK` to bypass runtime checks.

Other MAgent environments use `AdversarialPursuitPredator`,
`AdversarialPursuitPrey` or `BattleWrapper`, each accepting an independent seeded
opponent controller. `RandomController`, `PreyEscapeController`, `BattleController`
and `FrozenPolicyController` share `reset(seed, agents)`/`actions(...)` without
inheriting learning AgentGroups. Observation spaces describe the actual image,
channel-first or vector output. `PadAgentView` explicitly homogenizes variable
local observation/action spaces; padded actions remain illegal.
