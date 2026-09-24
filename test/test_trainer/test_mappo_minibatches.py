import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from test.test_trainer import test_mappo_trainer as fixtures
from marlite.trainer.trainer_worker.mappo_worker import MAPPOWorker
from marlite.util.action_distribution import masked_categorical


class TestMAPPOMinibatches(unittest.TestCase):
    def test_every_batch_updates_and_worker_matches(self):
        fixture = fixtures.TestMAPPOTrainer()
        fixture.setUp()
        fixture.config["trainer"]["train_device"] = "cpu"
        with tempfile.TemporaryDirectory() as directory:
            trainer = fixture._create_trainer(directory)
            batches = []
            for size in (4, 4, 2):
                batch = {
                    "observations": torch.randn(size, 2, 3, 18),
                    "states": torch.randn(size, 2, 54),
                    "next_states": torch.randn(size, 2, 54),
                    "alive_mask": torch.ones(size, 2, 3, dtype=torch.bool),
                    "next_alive_mask": torch.ones(size, 2, 3, dtype=torch.bool),
                    "timestep_padding_mask": torch.zeros(size, 2, dtype=torch.bool),
                    "next_timestep_padding_mask": torch.zeros(size, 2, dtype=torch.bool),
                    "actions": torch.zeros(size, 2, 3, dtype=torch.long),
                    "rewards": torch.randn(size, 2, 3),
                    "terminations": torch.zeros(size, 2, 3, dtype=torch.bool),
                    "avail_actions": torch.ones(size, 2, 3, 5, dtype=torch.bool),
                    "all_log_probs": torch.zeros(size, 2, 3),
                }
                batch["avail_actions"][..., 2:] = False
                batches.append(batch)
            trainer._prepare_ppo_dataset = Mock(return_value=None)
            with patch("marlite.trainer.mappo_trainer.TrajectoryDataLoader", return_value=batches), \
                 patch.object(trainer.agent_optimizer, "step", wraps=trainer.agent_optimizer.step) as actor, \
                 patch.object(trainer.critic_optimizer, "step", wraps=trainer.critic_optimizer.step) as critic:
                trainer.learn(10, 4, times=2)
                self.assertEqual(actor.call_count, 6)
                self.assertEqual(critic.call_count, 6)

            # The worker runs on CPU here: no process group or GPU is needed
            # to verify that its PPO distribution matches single-device code.
            worker = MAPPOWorker.__new__(MAPPOWorker)
            for name in ("eval_agent_group", "eval_critic", "agent_optimizer",
                         "critic_optimizer", "gamma", "clip_epsilon",
                         "entropy_coef", "vf_coef", "max_grad_norm", "reward_aggr_mode"):
                setattr(worker, name, getattr(trainer, name))
            worker.device = torch.device("cpu")
            worker._reduce_agent_gradients = lambda: None
            worker._reduce_critic_gradients = lambda: None
            batch = batches[0]
            with torch.no_grad():
                logits = trainer.eval_agent_group(batch["observations"].transpose(1, 2),
                    batch["timestep_padding_mask"].unsqueeze(1).expand(-1, 3, -1),
                    batch["alive_mask"][:, -1])["action_logits"]
                batch["all_log_probs"][:, -1] = masked_categorical(
                    logits, batch["avail_actions"][:, -1]).log_prob(batch["actions"][:, -1])
            with patch.object(trainer.agent_optimizer, "step"), patch.object(trainer.critic_optimizer, "step"), \
                 patch("marlite.trainer.mappo_trainer.TrajectoryDataLoader", return_value=[batch]):
                single = trainer.learn(4, 4, times=1)
                distributed = worker.train_step(batch)
            for key in ("actor_loss", "critic_loss", "loss"):
                self.assertAlmostEqual(single[key], distributed[key], places=5)
