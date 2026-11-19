"""All three value objectives must preserve the full-batch gradient under Gloo."""

from copy import deepcopy

import pytest
import torch
import torch.distributed as dist

from marlite.data import partition_episode
from test.sequence_test_tools import run_torchrun
from test.test_sequence_value import make_value_learner, collect_values


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo unavailable")
def test_value_objectives_match_single_with_uneven_and_empty_ranks(tmp_path):
    expected = {}
    for algorithm in ("qmix", "qplex", "qtran"):
        learner = make_value_learner(algorithm)
        episodes = collect_values(learner)
        sources = [s for i, ep in episodes.items() for s in partition_episode(i, ep, length=2)]
        first = learner.update(episodes, sources[:3])
        second = learner.update(episodes, sources[:1])
        expected[algorithm] = (deepcopy(learner.state_dict()), first, second)
    run_torchrun(tmp_path, '''
import os
from pathlib import Path
import torch
import torch.distributed as dist
from marlite.data import partition_episode
from marlite.training.distributed import initialize_backend
from test.test_sequence_value import make_value_learner, collect_values

backend = initialize_backend("cpu", timeout_seconds=45)
results = {}
try:
    for algorithm in ("qmix", "qplex", "qtran"):
        learner = make_value_learner(algorithm)
        episodes = collect_values(learner)
        sources = [s for i, ep in episodes.items() for s in partition_episode(i, ep, length=2)]
        first = learner.update(episodes, sources[:3][backend.rank::2])
        second = learner.update(episodes, sources[:1][backend.rank::2])
        if backend.rank == 0:
            results[algorithm] = (learner.state_dict(), first, second)
    # Check the replay runtime as well as isolated objective gradients.
    from test.test_sequence_value import make_value_trainer
    root = Path(os.environ["TEST_RUN_DIR"])
    trainer = make_value_trainer("qtran", root / "run", validation_replay_ratio=.5, rollback_interval=1)
    trainer.run(1, train_episodes=2, eval_episodes=2, checkpoint_interval=1)
    assert len(trainer.replay.episodes) == 3
    trainer.save_checkpoint(root / "resume.pt")
    reference = trainer.train_iteration(2)
    weights = {k: v.clone() for k, v in trainer.learner.actor.state_dict().items()}
    restored = make_value_trainer("qtran", root / "restored", validation_replay_ratio=.5, rollback_interval=1)
    restored.load_checkpoint(root / "resume.pt")
    assert reference == restored.train_iteration(2)
    for key, value in weights.items():
        torch.testing.assert_close(restored.learner.actor.state_dict()[key], value, rtol=0, atol=0)
    if backend.rank == 0:
        torch.save(results, root / "result.pt")
finally:
    dist.destroy_process_group()
''')
    actual = torch.load(tmp_path / "result.pt", weights_only=True)
    for algorithm, (weights, first, second) in expected.items():
        current, one, two = actual[algorithm]
        for module in ("actor", "critic", "objective", "target_actor", "target_critic"):
            for key, value in weights[module].items():
                torch.testing.assert_close(current[module][key], value, rtol=1e-5, atol=1e-6)
        assert current["counters"] == weights["counters"]
        for reference, distributed in ((first, one), (second, two)):
            for name, value in reference.items():
                assert distributed[name] == pytest.approx(value, rel=1e-5, abs=1e-6)
