# Complementary pursuit environment

The C04 example uses complementary pursuit from the CTmH MAgent2 fork. The
revision in `pyproject.toml` supplies the native attack rules used by the
environment wrapper and auxiliary label provider. Select the normal MARLite
accelerator extra for the target machine.

On Apple Silicon, building MAgent2 may require an OpenMP-enabled Clang. If
PyTorch and the system Clang both load their own OpenMP runtime, point the
loader to PyTorch's copy before launching Python so workers inherit it.

```sh
TORCH_LIB="$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"] + "/torch/lib")')"
DYLD_LIBRARY_PATH="$TORCH_LIB" .venv/bin/python main.py train --config=examples/distilled_group_communication_mappo.yaml
```
