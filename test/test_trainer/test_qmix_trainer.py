import unittest
import yaml
from copy import deepcopy
import tempfile
import torch
import pygame

from marlite.trainer import TrainerConfig


class TestQMixTrainer(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load("""
agent_group:
  type: "QMIX"
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
  type: "QMixer"
  model:
    model_type: QMixModel
    state_shape: 54
    input_dim: 3
    qmix_hidden_dim: 32
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
  type: "QMIX"
  gamma: 0.95
  eval_epsilon: 0.01
  workdir: "./test/results/replace_by_tempfile"
  train_device: "cpu"

  epsilon_scheduler:
    type: "logarithmic"
    start_value: 1.0
    end_value: 0.05
    decay_steps: 10

  sample_ratio_scheduler:
    type: "linear"
    start_value: 16
    end_value: 16
    decay_steps: 10

  train_args:
    epochs: 1
    target_first_metric: 100
    rollback_interval: 1
    batch_size: 8
    learning_times_per_epoch: 1
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
            origin_critic_params = deepcopy(self.trainer.target_critic.state_dict())
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=1)
            self.trainer.update_target_model_params()
            critic_params = self.trainer.target_critic.state_dict()

            for w1, w2 in zip(critic_params.values(), origin_critic_params.values()):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))

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
            best_metrics = self.trainer.train(epochs=2, target_first_metric=5)

    def test_target_update(self):

        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self._create_trainer(temp_dir)

            for param in self.trainer.eval_critic.parameters():
                torch.nn.init.ones_(param)
            for param in self.trainer.target_critic.parameters():
                torch.nn.init.zeros_(param)

            for (_, fe), (_, model) in zip(
                self.trainer.eval_agent_group.feature_extractors.items(),
                self.trainer.eval_agent_group.models.items(),
            ):
                for param in fe.parameters():
                    torch.nn.init.ones_(param)
                for param in model.parameters():
                    torch.nn.init.ones_(param)

            for (_, fe), (_, model) in zip(
                self.trainer.target_agent_group.feature_extractors.items(),
                self.trainer.target_agent_group.models.items(),
            ):
                for param in fe.parameters():
                    torch.nn.init.zeros_(param)
                for param in model.parameters():
                    torch.nn.init.zeros_(param)

            original_target_critic_params = deepcopy(
                self.trainer.target_critic.state_dict()
            )
            original_target_agent_group_params = deepcopy(
                self.trainer.target_agent_group.state_dict()
            )

            self.trainer.update_target_model_params()

            self.assertEqual(
                self.trainer.target_critic._modules.keys(),
                self.trainer.eval_critic._modules.keys(),
            )
            new_target_critic_params = deepcopy(self.trainer.target_critic.state_dict())
            for name in new_target_critic_params:
                self.assertTrue(
                    torch.equal(
                        new_target_critic_params[name],
                        self.trainer.eval_critic.state_dict()[name],
                    )
                )
                self.assertFalse(
                    torch.equal(
                        new_target_critic_params[name],
                        original_target_critic_params[name],
                    )
                )

            new_target_agent_group_params = deepcopy(
                self.trainer.target_agent_group.state_dict()
            )
            eval_agent_group_state_dict = self.trainer.eval_agent_group.state_dict()
            for name in original_target_agent_group_params:
                self.assertTrue(
                    torch.equal(
                        new_target_agent_group_params[name],
                        eval_agent_group_state_dict[name],
                    )
                )
                self.assertFalse(
                    torch.equal(
                        new_target_agent_group_params[name],
                        original_target_agent_group_params[name],
                    )
                )

            for param in self.trainer.eval_critic.parameters():
                torch.nn.init.normal_(param)
            for name in new_target_critic_params:
                self.assertFalse(
                    torch.equal(
                        self.trainer.eval_critic.state_dict()[name],
                        self.trainer.target_critic.state_dict()[name],
                    )
                )
                self.assertFalse(
                    torch.equal(
                        new_target_critic_params[name],
                        self.trainer.eval_critic.state_dict()[name],
                    )
                )

            for (_, fe), (_, model) in zip(
                self.trainer.eval_agent_group.feature_extractors.items(),
                self.trainer.eval_agent_group.models.items(),
            ):
                for param in fe.parameters():
                    torch.nn.init.normal_(param)
                for param in model.parameters():
                    torch.nn.init.normal_(param)

            new_target_agent_group_params = deepcopy(
                self.trainer.target_agent_group.state_dict()
            )
            for name in original_target_agent_group_params:
                self.assertFalse(
                    torch.equal(
                        new_target_agent_group_params[name],
                        eval_agent_group_state_dict[name],
                    )
                )
                self.assertFalse(
                    torch.equal(
                        new_target_agent_group_params[name],
                        original_target_agent_group_params[name],
                    )
                )

    def test_distributed_data_parallel(self):
        """Test DistributedDataParallel training with multiple devices."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping multi-GPU test")

        gpu_count = torch.cuda.device_count()
        if gpu_count < 2:
            self.skipTest(f"Need at least 2 GPUs for multi-GPU test, found {gpu_count}")

        config = deepcopy(self.config)
        config["rollout"]["episode_limit"] = 2
        config["replay_buffer"]["capacity"] = 2
        device_list = [f"cuda:{i}" for i in range(gpu_count)]
        config["trainer"]["train_device"] = device_list

        with tempfile.TemporaryDirectory() as temp_dir:
            config["trainer"]["workdir"] = temp_dir
            trainer_config = TrainerConfig(config)
            self.trainer = trainer_config.create_trainer()

            self.assertTrue(self.trainer.use_multi_gpu)
            self.assertEqual(len(self.trainer.device_list), gpu_count)
            self.assertIsNotNone(self.trainer.worker_group)
            self.assertEqual(self.trainer.worker_group.world_size, gpu_count)

            origin_critic_params = deepcopy(self.trainer.target_critic.state_dict())
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=1)
            self.trainer.update_target_model_params()
            critic_params = self.trainer.target_critic.state_dict()

            for w1, w2 in zip(critic_params.values(), origin_critic_params.values()):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))

    def test_torch_compile(self):
        config = deepcopy(self.config)
        config["replay_buffer"]["capacity"] = 2
        config["trainer"]["compile_models"] = True

        with tempfile.TemporaryDirectory() as temp_dir:
            config["trainer"]["workdir"] = temp_dir
            trainer_config = TrainerConfig(config)
            self.trainer = trainer_config.create_trainer()
            origin_critic_params = deepcopy(self.trainer.target_critic.state_dict())
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=1)
            self.trainer.update_target_model_params()
            critic_params = self.trainer.target_critic.state_dict()

            for w1, w2 in zip(critic_params.values(), origin_critic_params.values()):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))


if __name__ == "__main__":
    unittest.main()
