"""The complete QMPD stages use the same learner/rank reduction path as MAPPO."""

from copy import deepcopy

import pytest
import torch
import torch.distributed as dist

from test.sequence_test_tools import run_torchrun
from test.test_sequence_distillation import distillation_trainer, distillation_probe


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo unavailable")
def test_qmpd_stages_match_single_and_resume_in_two_ranks(tmp_path):
    cases = [
        dict(relation=False, replacement="all"),
        dict(relation=True, replacement="progressive"),
        dict(relation=True, online_weight=0.0),
        dict(offline_enabled=False),
    ]
    expected = []
    for case in cases:
        trainer = distillation_trainer(**case)
        trainer.train_iteration(2)
        expected.append(
            (
                deepcopy(trainer.learner.state_dict()),
                deepcopy(trainer.offline_metrics),
                distillation_probe(trainer.learner.actor),
            )
        )
    run_torchrun(
        tmp_path,
        """
import os
from pathlib import Path
from copy import deepcopy
import torch
import torch.distributed as dist
from marlite.training.distributed import initialize_backend
from test.test_sequence_distillation import distillation_trainer, distillation_probe
from test.test_sequence_mappo import assert_tree_equal

backend = initialize_backend("cpu", timeout_seconds=60)
root = Path(os.environ["TEST_RUN_DIR"])
cases = [dict(relation=False, replacement="all"),
         dict(relation=True, replacement="progressive"),
         dict(relation=True, online_weight=0.),
         dict(offline_enabled=False)]
results = []
try:
    for index, case in enumerate(cases):
        trainer = distillation_trainer(root / str(index), **case)
        if trainer.offline_enabled:
            trainer.run_offline(max_updates=2)
            trainer.save_checkpoint(root / f"offline_{index}.pt")
            restored = distillation_trainer(root / f"restore_{index}", **case)
            restored.load_checkpoint(root / f"offline_{index}.pt")
            trainer.run_offline()
            restored.run_offline()
            assert trainer.offline_metrics == restored.offline_metrics
            assert_tree_equal(trainer.learner.state_dict(), restored.learner.state_dict())
        trainer.train_iteration(2)
        if backend.rank == 0:
            results.append((deepcopy(trainer.learner.state_dict()), deepcopy(trainer.offline_metrics),
                            distillation_probe(trainer.learner.actor)))
    if backend.rank == 0:
        torch.save(results, root / "result.pt")
finally:
    dist.destroy_process_group()
""",
        timeout=150,
    )
    actual = torch.load(tmp_path / "result.pt", weights_only=True)

    def assert_close_tree(first, second, path="root"):
        if isinstance(first, torch.Tensor):
            try:
                if path.endswith("in_proj_bias"):
                    # A shared key bias adds the same constant to every key's
                    # attention score, so softmax cancels it exactly. FP32 gives
                    # tiny residual gradients; Adam can amplify these in this
                    # unidentifiable slice. Check all identifiable parameters,
                    # optimizer statistics and actual policy probe outputs.
                    a_q, a_k, a_v = first.chunk(3)
                    b_q, b_k, b_v = second.chunk(3)
                    torch.testing.assert_close(a_q, b_q, rtol=3e-5, atol=3e-6)
                    torch.testing.assert_close(a_v, b_v, rtol=3e-5, atol=3e-6)
                    assert torch.isfinite(a_k).all() and torch.isfinite(b_k).all()
                else:
                    torch.testing.assert_close(first, second, rtol=3e-5, atol=3e-6)
            except AssertionError as error:
                raise AssertionError(path) from error
        elif isinstance(first, dict):
            assert first.keys() == second.keys()
            for key in first:
                assert_close_tree(first[key], second[key], path + "/" + str(key))
        elif isinstance(first, (list, tuple)):
            assert len(first) == len(second)
            for index, (a, b) in enumerate(zip(first, second)):
                assert_close_tree(a, b, path + "/" + str(index))
        elif isinstance(first, float):
            assert second == pytest.approx(first, rel=3e-5, abs=3e-6)
        else:
            assert first == second

    assert_close_tree(expected, actual)
