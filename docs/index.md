# MARLite

MARLite is a Python toolkit for research on multi-agent reinforcement learning.
It supports training from recurrent trajectories, configurable communication
between agents, and reproducible evaluation.

- [Run your first experiment](quickstart.md) on a small MPE2 scenario.
- [Configure an experiment](sequence_configuration.md) with MAPPO, QMIX, QPLEX,
  QTRAN, or QMPD.
- [Write a custom component](custom_components.md) and select it from YAML.
- [Look up the Python interfaces](api.md) for trajectories, models, and experiments.

**QMPD** is short for **Quantized Messaging via Policy Distillation**. Its full
method name is *Fixed-Length Quantized Message Exchange Between Agents via Policy
Distillation*. The [QMPD guide](group_communication.md) covers the teacher,
student, and offline training stages.

Single-device training and single-node multi-GPU training use the same
experiment recipes. See the [migration and validation record](trajectory_refactor.md)
for the environments and device configurations used in verification.
