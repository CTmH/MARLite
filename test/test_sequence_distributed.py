"""Actual torchrun/Gloo training, including empty-rank and uneven-mask batches."""


import pytest
import torch
import torch.distributed as dist

from test.test_sequence_mappo import TinyParallel, make_trainer
from test.sequence_test_tools import run_torchrun


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo unavailable")
@pytest.mark.parametrize("batch_sequences", [1, 3])
def test_torchrun_matches_single_global_batches(tmp_path, batch_sequences):
    trainer = make_trainer(factory=lambda: TinyParallel(death=True), max_steps=5)
    trainer.batch_sequences = batch_sequences
    trainer.run(1, train_episodes=2, eval_episodes=2, checkpoint_interval=None)
    expected = trainer.learner.state_dict()
    code = '''
import os
from pathlib import Path
import torch
import torch.distributed as dist
from marlite.training.distributed import initialize_backend
from test.test_sequence_mappo import TinyParallel, make_trainer

backend = initialize_backend("cpu", timeout_seconds=45)
root = Path(os.environ["TEST_RUN_DIR"])
try:
    # This also checks missing gradients and globally unused optimizer parameters.
    p = torch.nn.Parameter(torch.tensor([2.]))
    private = torch.nn.Parameter(torch.tensor([1.]))
    unused = torch.nn.Parameter(torch.tensor([3.]))
    x = torch.tensor([1., 2.]) if backend.rank == 0 else torch.tensor([3.])
    count = backend.sum(torch.tensor(float(len(x))))
    objective = (p * x).square().sum() / count
    if backend.rank == 1:
        objective = objective + private.square().sum()
    objective.backward()
    backend.reduce_gradients([p, private, unused])
    torch.testing.assert_close(p.grad, torch.tensor([56. / 3.]))
    torch.testing.assert_close(private.grad, torch.tensor([2.]))
    assert unused.grad is None
    trainer = make_trainer(factory=lambda: TinyParallel(death=True), max_steps=5,
                           workdir=root / "run")
    trainer.batch_sequences = int(os.environ["TEST_BATCH_SEQUENCES"])
    trainer.run(1, train_episodes=2, eval_episodes=2, checkpoint_interval=1)
    trainer.save_checkpoint(root / "resume.pt")
    expected_next = trainer.train_iteration(2)
    expected_state = {k: p.detach().clone() for k, p in trainer.learner.actor.state_dict().items()}
    restored = make_trainer(factory=lambda: TinyParallel(death=True), max_steps=5,
                            workdir=root / "restored")
    restored.batch_sequences = trainer.batch_sequences
    restored.load_checkpoint(root / "resume.pt")
    actual_next = restored.train_iteration(2)
    assert expected_next == actual_next
    for key, value in restored.learner.actor.state_dict().items():
        torch.testing.assert_close(value, expected_state[key], rtol=0, atol=0)
    # Reload first-iteration state for comparison with the single-device parent.
    trainer.load_checkpoint(root / "resume.pt")
    if backend.rank == 0:
        torch.save(trainer.learner.state_dict(), root / "result.pt")
        assert not list((root / "run" / "rollouts").glob("round_*"))
finally:
    dist.destroy_process_group()
'''
    run_torchrun(tmp_path, code, TEST_BATCH_SEQUENCES=str(batch_sequences))
    actual = torch.load(tmp_path / "result.pt", weights_only=True)
    for component in ("actor", "critic"):
        for name, tensor in expected[component].items():
            torch.testing.assert_close(actual[component][name], tensor, rtol=1e-5, atol=1e-6)
    assert actual["actor_updates"] == expected["actor_updates"]
    assert actual["critic_updates"] == expected["critic_updates"]
    assert len((tmp_path / "run" / "events.jsonl").read_text().splitlines()) == 1
