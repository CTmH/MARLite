"""Measure fused vs per-step sequence computation, including backward work."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import statistics
import time

import torch

from marlite.algorithm.temporal import CausalAttention, RecurrentGRU


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(2)
    shape = (4, 32, 32, 64)
    x = torch.randn(*shape, device=device)
    alive = torch.ones(shape[:3], device=device, dtype=torch.bool)
    alive[:, :, -4:] = False
    alive[:, 16:20, 0] = False
    reset = torch.zeros_like(alive)
    reset[:, 0] = True
    reset[:, 20, 0] = True
    report = {"device": str(device), "shape": shape, "torch": torch.__version__, "models": {}}

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

    for kind, model in (
        ("gru", RecurrentGRU(64, 64)),
        ("attention", CausalAttention(64, 64, 4, 32)),
    ):
        paths = {}
        for path in ("step_reference", "fused_sequence"):
            network = deepcopy(model).to(device)
            initial = network.initial_state(shape[0], shape[2], device, x.dtype)
            times = []
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            for repeat in range(args.repetitions + 2):
                network.zero_grad(set_to_none=True)
                inputs = x.detach().requires_grad_()
                sync()
                begin = time.perf_counter()
                if path == "fused_sequence":
                    output = network.forward_sequence(inputs, initial, alive, reset).features
                else:
                    state, rows = initial, []
                    for step in range(shape[1]):
                        result = network.step(
                            inputs[:, step], state, alive[:, step], reset[:, step]
                        )
                        rows.append(result.features)
                        state = result.state
                    output = torch.stack(rows, 1)
                output.square().mean().backward()
                sync()
                if repeat >= 2:
                    times.append(1000 * (time.perf_counter() - begin))
            paths[path] = {
                "forward_backward_ms_median": statistics.median(times),
                "cuda_peak_allocated_bytes": (
                    torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
                ),
            }
        paths["speedup"] = (
            paths["step_reference"]["forward_backward_ms_median"]
            / paths["fused_sequence"]["forward_backward_ms_median"]
        )
        report["models"][kind] = paths
    # Analytic state storage only; observation/transition storage is additional.
    frames, agents, context, width, interval = 201, 100, 128, 128, 32
    per_snapshot = agents * context * (width * 4 + 1)
    report["attention_cache_storage_example"] = {
        "frames": frames,
        "agents": agents,
        "context": context,
        "hidden": width,
        "interval": interval,
        "dense_bytes": frames * per_snapshot,
        "sparse_bytes": ((frames - 1) // interval + 1) * per_snapshot,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
