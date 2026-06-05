import unittest
import os
import yaml
from copy import deepcopy
import tempfile
import torch
import pygame

from marlite.trainer import TrainerConfig


class TestAEGCMAPPOTrainer(unittest.TestCase):
    def setUp(self):
        self.config_path = "test/config/ae_gc_mappo_default.yaml"
        with open(self.config_path, "r") as file:
            self.config = yaml.safe_load(file)
        self.config["trainer"]["train_args"]["iterations"] = 2
        self.config["rollout"]["n_episodes"] = 2
        self.config["rollout"]["n_eval_episodes"] = 2
        self.config["rollout"]["episode_limit"] = 20
        self.config["replay_buffer"]["capacity"] = 5
        self.trainer_config = TrainerConfig(self.config)

    def test_collect_experience(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, "logs")
            self.trainer.checkpointdir = os.path.join(
                self.trainer.workdir, "checkpoints"
            )
            self.trainer.collect_experience(0.9)
            self.assertNotEqual(len(self.trainer.replaybuffer.buffer), 0)

    def test_learn(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, "logs")
            self.trainer.checkpointdir = os.path.join(
                self.trainer.workdir, "checkpoints"
            )
            origin_actor_params = deepcopy(
                self.trainer.eval_agent_group.state_dict()
            )
            origin_critic_params = deepcopy(
                self.trainer.eval_critic.state_dict()
            )
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=1)

            actor_params = self.trainer.eval_agent_group.state_dict()
            for key in actor_params:
                if actor_params[key].requires_grad:
                    self.assertFalse(
                        torch.equal(
                            actor_params[key], origin_actor_params[key]
                        ),
                        f"Actor param {key} did not change after learning",
                    )

            critic_params = self.trainer.eval_critic.state_dict()
            for key in critic_params:
                if critic_params[key].requires_grad:
                    self.assertFalse(
                        torch.equal(
                            critic_params[key], origin_critic_params[key]
                        ),
                        f"Critic param {key} did not change after learning",
                    )

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
        pygame.font.init()
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, "logs")
            self.trainer.checkpointdir = os.path.join(
                self.trainer.workdir, "checkpoints"
            )
            result = self.trainer.evaluate()
            best_metrics = self.trainer.train(
                iterations=2, target_first_metric=5
            )


if __name__ == "__main__":
    unittest.main()