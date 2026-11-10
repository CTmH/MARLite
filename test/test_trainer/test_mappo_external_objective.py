import tempfile
from unittest.mock import Mock, patch
import pytest
import torch
from test.test_trainer import test_mappo_trainer as fixtures


def batch():
    n = 4
    return {
        "observations": torch.randn(n, 2, 3, 18),
        "states": torch.randn(n, 2, 54),
        "next_states": torch.randn(n, 2, 54),
        "alive_mask": torch.ones(n, 2, 3, dtype=torch.bool),
        "next_alive_mask": torch.ones(n, 2, 3, dtype=torch.bool),
        "timestep_padding_mask": torch.zeros(n, 2, dtype=torch.bool),
        "next_timestep_padding_mask": torch.zeros(n, 2, dtype=torch.bool),
        "actions": torch.zeros(n, 2, 3, dtype=torch.long),
        "rewards": torch.randn(n, 2, 3),
        "terminations": torch.zeros(n, 2, 3, dtype=torch.bool),
        "avail_actions": torch.ones(n, 2, 3, 5, dtype=torch.bool),
        "all_log_probs": torch.zeros(n, 2, 3),
    }


def test_auxiliary_objective_affects_actor_only():
    fixture = fixtures.TestMAPPOTrainer()
    fixture.setUp()
    fixture.config["trainer"]["train_device"] = "cpu"
    with tempfile.TemporaryDirectory() as directory:
        trainer = fixture._create_trainer(directory)
        trainer._prepare_ppo_dataset = Mock(return_value=None)
        b = batch()
        gradients = []
        critic_gradients = []
        calls = []
        def objective(model):
            calls.append(1)
            return sum(p.square().sum() for p in model.parameters())
        for callback in (None, objective):
            trainer.actor_auxiliary_objective = callback
            with patch.object(trainer.agent_optimizer, "step"), patch.object(trainer.critic_optimizer, "step"), \
                 patch("torch.nn.utils.clip_grad_norm_"), \
                 patch("marlite.trainer.mappo_trainer.TrajectoryDataLoader", return_value=[b]):
                trainer.learn(4, 4, times=1)
            gradients.append(torch.cat([p.grad.flatten() for p in trainer.eval_agent_group.parameters() if p.grad is not None]))
            critic_gradients.append(torch.cat([p.grad.flatten() for p in trainer.eval_critic.parameters() if p.grad is not None]))
        assert calls == [1]
        assert not torch.allclose(*gradients)
        torch.testing.assert_close(*critic_gradients)
        trainer.actor_auxiliary_objective = lambda _: torch.tensor(float("nan"))
        with patch("marlite.trainer.mappo_trainer.TrajectoryDataLoader", return_value=[b]):
            with pytest.raises(ValueError, match="finite scalar"):
                trainer.learn(4, 4, times=1)
