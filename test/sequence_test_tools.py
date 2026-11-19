"""Run isolated torchrun tests with bounded time and whole-process cleanup."""

import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest


def run_torchrun(directory, code, **variables):
    script = directory / "distributed_run.py"
    script.write_text(code, encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               TEST_RUN_DIR=str(directory), PYTHONDONTWRITEBYTECODE="1", **variables)
    command = [sys.executable, "-m", "torch.distributed.run", "--rdzv-backend=c10d",
               "--rdzv-endpoint=127.0.0.1:0", "--local-addr=127.0.0.1",
               "--nnodes=1", "--nproc-per-node=2", str(script)]
    process = subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        output, _ = process.communicate(timeout=75)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            output, _ = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
        pytest.fail("torchrun timed out:\n" + output)
    assert process.returncode == 0, output
