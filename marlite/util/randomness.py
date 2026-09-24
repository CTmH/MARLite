"""Stable experiment streams without relying on process IDs or Python hash()."""

import os
import random
from contextlib import contextmanager

import numpy as np
import torch


# Stable stream identifiers; changing these changes the experiment's random draws.
MODEL_STREAM = 0
SSL_MODEL_STREAM = 1
VALUE_MODEL_STREAM = 2
TRAIN_STREAM = 3
WORKER_STREAM = 4
ROLLOUT_STREAM = 5
EVALUATION_STREAM = 6


def derive_seed(seed, *coordinates):
    """Return a NumPy/environment-compatible uint32, or preserve opt-out None."""
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer or null")
    return int(np.random.SeedSequence([seed, *coordinates]).generate_state(1)[0])


def seed_everything(seed):
    """Seed process-global RNGs; libraries with private generators need a seed too."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def configure_randomness(seed=None, deterministic=False, stream=MODEL_STREAM):
    """Configure before model/device creation. Defaults do not alter global state.

    Strict deterministic mode intentionally raises for unsupported operations.
    It is not a promise of bitwise equality across hardware/library versions.
    """
    if not isinstance(deterministic, bool):
        raise ValueError("deterministic must be a boolean")
    derived = derive_seed(seed, stream)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if deterministic or seed is not None:
        torch.use_deterministic_algorithms(deterministic)
        torch.backends.cudnn.deterministic = deterministic
        if deterministic:
            torch.backends.cudnn.benchmark = False
    seed_everything(derived)


@contextmanager
def preserve_rng_state():
    """Run an optional diagnostic without consuming training random draws."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    mps_state = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)
