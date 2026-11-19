"""Real two-rank task mixing must equal one global sequence update."""

from copy import deepcopy

import pytest
import torch
import torch.distributed as dist

from test.sequence_test_tools import run_torchrun
from test.test_sequence_auxiliary import auxiliary_learner, auxiliary_data


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo unavailable")
def test_communication_ssl_matches_global_batches_and_resume(tmp_path):
    expected = {}
    for algorithm in ("mappo", "qmix", "qplex", "qtran"):
        for mixer in ("weighted", "pit", "ema", "pcgrad"):
            learner, provider = auxiliary_learner(algorithm, mixer=mixer)
            episodes, sources = auxiliary_data(learner, provider)
            # Uneven partition first, then an empty rank. Noise follows frame IDs.
            first = learner.update(episodes, sources[:3])
            second = learner.update(episodes, sources[:1])
            expected[(algorithm, mixer)] = (deepcopy(learner.state_dict()), first, second)
    run_torchrun(
        tmp_path,
        """
import os
from pathlib import Path
from copy import deepcopy
import torch
import torch.distributed as dist
from marlite.training.distributed import initialize_backend
from test.test_sequence_auxiliary import auxiliary_learner, auxiliary_data
from test.test_sequence_mappo import assert_tree_equal

backend = initialize_backend("cpu", timeout_seconds=60)
results = {}
try:
    for algorithm in ("mappo", "qmix", "qplex", "qtran"):
        for mixer in ("weighted", "pit", "ema", "pcgrad"):
            learner, provider = auxiliary_learner(algorithm, mixer=mixer)
            episodes, sources = auxiliary_data(learner, provider)
            first = learner.update(episodes, sources[:3][backend.rank::2])
            checkpoint = deepcopy(learner.state_dict())
            second = learner.update(episodes, sources[:1][backend.rank::2])
            restored, _ = auxiliary_learner(algorithm, mixer=mixer)
            restored.load_state_dict(checkpoint)
            actual = restored.update(episodes, sources[:1][backend.rank::2])
            assert actual == second
            assert_tree_equal(restored.state_dict(), learner.state_dict())
            if backend.rank == 0:
                results[(algorithm, mixer)] = (learner.state_dict(), first, second)
    if backend.rank == 0:
        torch.save(results, Path(os.environ["TEST_RUN_DIR"]) / "result.pt")
finally:
    dist.destroy_process_group()
""",
        timeout=150,
    )
    actual = torch.load(tmp_path / "result.pt", weights_only=True)

    def assert_close_tree(first, second):
        if isinstance(first, torch.Tensor):
            torch.testing.assert_close(first, second, rtol=3e-5, atol=2e-6)
        elif isinstance(first, dict):
            assert first.keys() == second.keys()
            for key in first:
                assert_close_tree(first[key], second[key])
        elif isinstance(first, (tuple, list)):
            assert len(first) == len(second)
            for a, b in zip(first, second):
                assert_close_tree(a, b)
        elif isinstance(first, float):
            assert second == pytest.approx(first, rel=3e-5, abs=2e-6)
        else:
            assert first == second

    for key in expected:
        assert_close_tree(expected[key], actual[key])
