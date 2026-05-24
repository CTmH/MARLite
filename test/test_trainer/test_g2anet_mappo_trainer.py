import unittest
import os
import yaml
from copy import deepcopy
import tempfile
import torch

from marlite.trainer import TrainerConfig


class TestG2ANetMAPPOTrainer(unittest.TestCase):
    def setUp(self):
        self.config_path = "test/config/g2anet_mappo_default.yaml"
        with open(self.config_path, "r") as file:
            self.config = yaml.safe_load(file)
        self.config["trainer"]["train_args"]["iterations"] = 2
        self.config["rollout"]["n_episodes"] = 2
        self.config["rollout"]["n_eval_episodes"] = 2
        self.config["rollout"]["episode_limit"] = 2
        self.config["replay_buffer"]["capacity"] = 2
        self.trainer_config = TrainerConfig(self.config)

    def test_collect_experience(self):
        """On-policy: calling evaluate() fills the replay buffer."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, "logs")
            self.trainer.checkpointdir = os.path.join(
                self.trainer.workdir, "checkpoints"
            )
            self.trainer.evaluate()
            self.assertNotEqual(len(self.trainer.replaybuffer.buffer), 0)

    def test_learn(self):
        """Verify parameters change after learning."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, "logs")
            self.trainer.checkpointdir = os.path.join(
                self.trainer.workdir, "checkpoints"
            )

            # Fill buffer with initial rollout
            self.trainer.evaluate()

            # Snapshot critic params before learning
            origin_critic = deepcopy(self.trainer.eval_critic.state_dict())
            origin_agent = deepcopy(self.trainer.eval_agent_group.state_dict())

            self.trainer.learn(sample_size=32, batch_size=8, times=1)

            critic_params = self.trainer.eval_critic.state_dict()
            agent_params = self.trainer.eval_agent_group.state_dict()

            for w1, w2 in zip(critic_params.values(), origin_critic.values()):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))
            for w1, w2 in zip(agent_params.values(), origin_agent.values()):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))

    def test_save_load_checkpoint(self):
        checkpoint = "test_checkpoint"
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, "logs")
            self.trainer.checkpointdir = os.path.join(
                self.trainer.workdir, "checkpoints"
            )
            self.trainer.save_current_model(checkpoint)
            self.trainer.load_checkpoint(checkpoint)

    def test_train(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, "logs")
            self.trainer.checkpointdir = os.path.join(
                self.trainer.workdir, "checkpoints"
            )
            result = self.trainer.evaluate()
            best_metrics = self.trainer.train(
                iterations=2, target_first_metric=100
            )

    def test_torch_compile(self):
        self.config_path = "test/config/g2anet_mappo_default.yaml"
        with open(self.config_path, "r") as file:
            self.config = yaml.safe_load(file)
        self.config["trainer"]["train_args"]["iterations"] = 2
        self.config["rollout"]["n_episodes"] = 2
        self.config["rollout"]["n_eval_episodes"] = 2
        self.config["rollout"]["episode_limit"] = 2
        self.config["replay_buffer"]["capacity"] = 2
        self.config["trainer"]["compile_models"] = True
        self.trainer_config = TrainerConfig(self.config)

        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, "logs")
            self.trainer.checkpointdir = os.path.join(
                self.trainer.workdir, "checkpoints"
            )
            self.trainer.evaluate()
            origin_critic = deepcopy(self.trainer.eval_critic.state_dict())
            self.trainer.learn(sample_size=32, batch_size=8, times=1)
            critic_params = self.trainer.eval_critic.state_dict()

            for w1, w2 in zip(critic_params.values(), origin_critic.values()):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))


if __name__ == "__main__":
    unittest.main()