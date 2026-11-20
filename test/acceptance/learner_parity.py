"""Bounded CPU/GPU acceptance; launch once locally or through torchrun.

The same global batches train on one, two or three devices. Each run also
checks same-topology exact continuation. This is not a research experiment.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

import torch
import torch.distributed as dist

from marlite.algorithm.sequence_agent_group import DecisionInput
from marlite.configuration.recipe import build_experiment
from marlite.training.distributed import initialize_backend
from test.test_configuration import minimal_config
from test.test_sequence_mappo import TinyParallel, assert_tree_equal


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def measure_iteration(trainer, backend):
    backend.barrier()
    device = backend.device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = perf_counter()
    metrics = trainer.train_iteration(2)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = perf_counter() - start
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    per_rank = backend.gather_objects({"seconds": elapsed, "peak_allocated_bytes": peak})
    return metrics, per_rank


def gradient_error(actual, expected):
    assert len(actual) == len(expected)
    worst = 0.0
    for actual_group, expected_group in zip(actual, expected, strict=True):
        assert len(actual_group) == len(expected_group)
        for gradient, reference in zip(actual_group, expected_group, strict=True):
            if gradient is None or reference is None:
                assert gradient is reference
                continue
            torch.testing.assert_close(gradient, reference, rtol=1e-4, atol=1e-5)
            worst = max(worst, float((gradient - reference).abs().max()))
    return worst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    backend = initialize_backend(args.device, timeout_seconds=120)
    sys.modules["acceptance_tiny_env"] = SimpleNamespace(parallel_env=TinyParallel)
    results = {}
    capture = None
    reduce_gradients = backend.reduce_gradients

    def record_gradients(parameters):
        parameters = tuple(dict.fromkeys(parameters))
        reduce_gradients(parameters)
        if backend.rank == 0 and capture is not None:
            capture.append(tuple(cpu_tree(parameter.grad) for parameter in parameters))

    backend.reduce_gradients = record_gradients
    try:
        for recipe in ("mappo", "qmix", "qplex", "qtran", "qmpd"):
            config = minimal_config("acceptance_tiny_env", recipe)
            config["schedulers"] = {
                role: {"scheduler": {"type": "StepLR", "params": {"step_size": 1, "gamma": 0.9}}}
                for role in ("actor", "critic")
            }
            experiment = build_experiment(config, backend=backend, workdir=args.workdir / recipe)
            capture = []
            experiment.run()
            gradients, capture = capture, None
            trainer = experiment.trainer
            snapshot = args.workdir / recipe / "acceptance.pt"
            trainer.save_checkpoint(snapshot)
            expected_metrics, per_rank = measure_iteration(trainer, backend)
            expected = deepcopy(trainer.learner.state_dict())
            trainer.load_checkpoint(snapshot)
            assert expected_metrics == trainer.train_iteration(2), recipe
            assert_tree_equal(expected, trainer.learner.state_dict())
            actor = trainer.learner.actor.eval()
            x = torch.linspace(-1.0, 1.0, 24, device=backend.device).reshape(1, 4, 2, 3)
            valid = torch.ones(1, 4, 2, dtype=torch.bool, device=backend.device)
            reset = torch.zeros_like(valid)
            reset[:, 0] = True
            context = (
                {"groups": torch.zeros_like(valid, dtype=torch.long)} if recipe == "qmpd" else {}
            )
            with torch.no_grad():
                output = actor(
                    DecisionInput(x, valid, reset, context),
                    actor.initial_state(1, 2, backend.device, x.dtype),
                ).values
            if backend.rank == 0:
                results[recipe] = {
                    "learner": cpu_tree(trainer.learner.state_dict()),
                    "output": output.cpu(),
                    "gradients": gradients,
                    "iteration_by_rank": per_rank,
                }
        if backend.rank == 0:
            args.workdir.mkdir(parents=True, exist_ok=True)
            torch.save(results, args.workdir / "result.pt")
            report = {
                "world_size": backend.world_size,
                "device": str(backend.device),
                "torch": torch.__version__,
                "hip": torch.version.hip,
                "same_topology_resume": "exact",
                "recipes": {},
            }
            reference = torch.load(args.reference, weights_only=True) if args.reference else None
            for recipe, result in results.items():
                evidence = {
                    "updates": result["learner"].get("updates", result["learner"].get("counters")),
                    "actor_finite": bool(torch.isfinite(result["output"]).all()),
                    "gradient_groups": len(result["gradients"]),
                    "iteration_by_rank": result["iteration_by_rank"],
                }
                if reference is not None:
                    other = reference[recipe]
                    evidence["gradient_max_abs_error"] = gradient_error(
                        result["gradients"], other["gradients"]
                    )
                    torch.testing.assert_close(
                        result["output"], other["output"], rtol=1e-4, atol=1e-5
                    )
                    evidence["output_max_abs_error"] = float(
                        (result["output"] - other["output"]).abs().max()
                    )
                    worst = 0.0
                    for component in ("actor", "critic"):
                        for name, parameter in result["learner"][component].items():
                            expected = other["learner"][component][name]
                            if not parameter.is_floating_point():
                                assert torch.equal(parameter, expected)
                                continue
                            if name.endswith("in_proj_bias"):
                                # Shared key bias cancels exactly in softmax; its
                                # floating residual is not an identifiable parameter.
                                pq, _, pv = parameter.chunk(3)
                                eq, _, ev = expected.chunk(3)
                                parameter, expected = torch.cat((pq, pv)), torch.cat((eq, ev))
                            torch.testing.assert_close(
                                parameter,
                                expected,
                                rtol=1e-4,
                                atol=1e-5,
                                msg=f"{recipe}/{component}/{name}",
                            )
                            worst = max(worst, float((parameter - expected).abs().max()))
                    evidence["identifiable_parameter_max_abs_error"] = worst
                report["recipes"][recipe] = evidence
            (args.workdir / "report.json").write_text(json.dumps(report, indent=2))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
