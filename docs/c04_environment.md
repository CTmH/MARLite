# C04 development environment

Use exactly one accelerator extra. The project ignores uv.lock; torch versions and the MAgent2 revision are explicitly pinned in pyproject.toml. Preserve an experiment copy of uv.lock for full transitive reproducibility.

## Apple Silicon (current MacPorts installation)

```sh
CC=/opt/local/bin/clang-mp-21 CXX=/opt/local/bin/clang++-mp-21 uv sync --extra mps --extra develop --extra analysis
```

MAgent2 requires OpenMP. The default Apple compiler did not locate it; the existing MacPorts Clang 21 builds the pinned native extension. Do not install a second compiler when the required one is already available.

PyTorch and MacPorts each supply libomp. For processes importing both torch and magent2, select the PyTorch copy before starting Python; do not use KMP_DUPLICATE_LIB_OK:

```sh
TORCH_LIB="$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"] + "/torch/lib")')"
DYLD_LIBRARY_PATH="$TORCH_LIB" NUMBA_CACHE_DIR=/tmp/marlite-numba .venv/bin/python main.py train --config=CONFIG.yaml
```

This is a local launch requirement, not a global shell setting. Recheck it after upgrading either native package. The same environment must reach spawned workers. Verified with torch 2.15.0 MPS forward/backward and complementary_pursuit_v1 reset/step (29 observation channels, 25 state channels). A complete training run has not been verified.

## iu1iu5-local

Read-only inventory: two AMD Radeon RX 7900 XT and one RX 7900 XTX; existing torch 2.14.0+rocm7.14, HIP 7.14.60850. Select rocm, not cuda:

```sh
uv sync --extra rocm --extra develop
```

The new project pin resolves to torch 2.15.0+rocm7.14. Server installation and training have not been performed; verify GPU execution and native MAgent2 after the authorized update. CUDA remains supported separately with torch 2.14.1 from the existing cu130 index; do not switch indexes solely to match the MPS version.

## Implemented first increment

GeneralHyperNetwork emits signed weights. QMixHyperNetwork applies the historical nonnegative transform. HyperNetwork remains a compatibility alias; existing state-dict parameter names are preserved. The shared group-communication agents, dual-algorithm trainers and new rollout fields are subsequent work, not implemented by this increment.
