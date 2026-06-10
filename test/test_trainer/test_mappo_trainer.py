import unittest
import yaml
from copy import deepcopy
import tempfile
import torch
import pygame

from marlite.trainer import TrainerConfig


class TestMAPPOTrainer(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load("""
agent_group:
  type: "MAPPO"
  agent_list:
    agent_0: model1
    agent_1: model1
    agent_2: model1
  models:
    model1:
      feature_extractor:
        model_type: "Identity"
      model:
        model_type: "RNN"
        input_shape: 18
        rnn_hidden_dim: 32
        rnn_layers: 1
        output_shape: 5
  optimizer:
    type: "Adam"
    lr: 0.0005
    weight_decay: 0.0001

environment:
  module_name: "mpe2"
  env_name: "simple_spread_v3"

critic:
  type: "MAPPOCritic"
  model:
    model_type: "Custom"
    layers:
      - type: "Linear"
        in_features: 54
        out_features: 128
      - type: "ReLU"
      - type: "Linear"
        in_features: 128
        out_features: 128
      - type: "ReLU"
      - type: "Linear"
        in_features: 128
        out_features: 1
  feature_extractor:
    model_type: "Identity"
  optimizer:
    type: "Adam"
    lr: 0.0005
    weight_decay: 0.0001

rollout:
  manager_type: "multi-process"
  worker_type: "multi-process"
  n_workers: 1
  n_episodes: 1
  n_eval_episodes: 1
  traj_len: 7
  episode_limit: 20
  device: "cpu"

replay_buffer:
  type: "Normal"
  capacity: 5
  traj_len: 7

analyzer:
  type: "default"

trainer:
  type: "MAPPO"
  gamma: 0.99
  workdir: "./test/results/replace_by_tempfile"
  train_device: "cpu"
  clip_epsilon: 0.2
  gae_lambda: 0.95
  entropy_coef: 0.01
  vf_coef: 0.5
  max_grad_norm: 5.0

  sample_ratio_scheduler:
    type: "linear"
    start_value: 16
    end_value: 16
    decay_steps: 10

  train_args:
    iterations: 1
    target_first_metric: -9999
    batch_size: 8
    learning_times_per_iteration: 4
""")

        if torch.cuda.is_available():
            self.config["trainer"]["train_device"] = "cuda"
            self.config["rollout"]["device"] = "cuda"

    def _create_trainer(self, temp_dir):
        self.config['trainer']['workdir'] = temp_dir
        trainer_config = TrainerConfig(self.config)
        return trainer_config.create_trainer()

    def test_collect_experience(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self._create_trainer(temp_dir)
            self.trainer.collect_experience(0.9)
            self.assertNotEqual(len(self.trainer.replaybuffer.buffer), 0)

    def test_learn(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self._create_trainer(temp_dir)
            origin_actor_params = deepcopy(self.trainer.eval_agent_group.state_dict())
            origin_critic_params = deepcopy(self.trainer.eval_critic.state_dict())
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=1)

            actor_params = self.trainer.eval_agent_group.state_dict()
            for key in actor_params:
                if actor_params[key].requires_grad:
                    self.assertFalse(
                        torch.equal(actor_params[key], origin_actor_params[key]),
                        f"Actor param {key} did not change after learning",
                    )

            critic_params = self.trainer.eval_critic.state_dict()
            for key in critic_params:
                if critic_params[key].requires_grad:
                    self.assertFalse(
                        torch.equal(critic_params[key], origin_critic_params[key]),
                        f"Critic param {key} did not change after learning",
                    )

    def test_save_load_checkpoint(self):
        checkpoint = "test_checkpoint"
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self._create_trainer(temp_dir)
            self.trainer.save_current_model(checkpoint)
            self.trainer.load_checkpoint(checkpoint)

    def test_train(self):
        pygame.font.init()
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self._create_trainer(temp_dir)
            result = self.trainer.evaluate()
            best_metrics = self.trainer.train(
                iterations=2, target_first_metric=5
            )

    def test_distributed_data_parallel(self):
        """Test multi-GPU training with multiple devices."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping multi-GPU test")

        gpu_count = torch.cuda.device_count()
        if gpu_count < 2:
            self.skipTest(
                f"Need at least 2 GPUs for multi-GPU test, found {gpu_count}"
            )

        config = deepcopy(self.config)
        device_list = [f"cuda:{i}" for i in range(gpu_count)]
        config["trainer"]["train_device"] = device_list

        with tempfile.TemporaryDirectory() as temp_dir:
            config["trainer"]["workdir"] = temp_dir
            trainer_config = TrainerConfig(config)
            self.trainer = trainer_config.create_trainer()

            self.assertTrue(self.trainer.use_multi_gpu)
            self.assertEqual(len(self.trainer.device_list), gpu_count)
            self.assertIsNotNone(self.trainer.worker_group)
            self.assertEqual(
                self.trainer.worker_group.world_size, gpu_count
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
                        f"Actor param {key} did not change after multi-GPU learning",
                    )

            critic_params = self.trainer.eval_critic.state_dict()
            for key in critic_params:
                if critic_params[key].requires_grad:
                    self.assertFalse(
                        torch.equal(
                            critic_params[key], origin_critic_params[key]
                        ),
                        f"Critic param {key} did not change after multi-GPU learning",
                    )


if __name__ == "__main__":
    unittest.main()
