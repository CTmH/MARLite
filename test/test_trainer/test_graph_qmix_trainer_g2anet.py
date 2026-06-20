import unittest
import yaml
from copy import deepcopy
import tempfile
import torch

from marlite.trainer import TrainerConfig


class TestG2ANetQMIXTrainer(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load("""
agent_group:
  type: "G2ANet"
  agent_list:
    predator_0: model1
    predator_1: model1
    predator_2: model1
    predator_3: model1
    predator_4: model1
    predator_5: model1
    predator_6: model1
    predator_7: model1
    predator_8: model1
    predator_9: model1
    predator_10: model1
    predator_11: model1
    predator_12: model1
    predator_13: model1
    predator_14: model1
    predator_15: model1
    predator_16: model1
    predator_17: model1
    predator_18: model1
    predator_19: model1
    predator_20: model1
    predator_21: model1
    predator_22: model1
    predator_23: model1
    predator_24: model1
  models:
    model1:
      feature_extractor:
        model_type: "Custom"
        layers:
        - type: Conv2d
          in_channels: 29
          out_channels: 8
          kernel_size: 1
          stride: 1
          padding: 0
        - type: LeakyReLU
        - type: Conv2d
          in_channels: 8
          out_channels: 1
          kernel_size: 3
          stride: 1
          padding: 0
        - type: BatchNorm2d
          num_features: 1
        - type: LeakyReLU
        - type: Flatten
      encoder:
        model_type: "Custom"
        layers:
        - type: Linear
          in_features: 64
          out_features: 64
        - type: Flatten
        - type: LeakyReLU
      decoder:
        model_type: "Custom"
        layers:
        - type: LeakyReLU
        - type: Linear
          in_features: 128
          out_features: 13
  graph_builder:
    type: "G2ANet"
    n_agents: 25
    add_self_loop: false
    input_dim: 64
    hidden_dim: 64
  graph_model:
    model_type: "MatrixGCN"
    input_dim: 64
    hidden_dim: 64
    output_dim: 64
  optimizer:
    type: "Adam"
    lr: 0.0002
    weight_decay: 0.00005

environment:
  module_name: "magent2.environments"
  env_name: "adversarial_pursuit_v4"
  env_params:
    tag_penalty: 0.0
    extra_features: true
  wrapper:
    type: adversarial_pursuit_predator
    opponent_agent_group:
      type: "MAgentPrey"
      agent_list:
        prey_0: random1
        prey_1: random1
        prey_2: random1
        prey_3: random1
        prey_4: random1
        prey_5: random1
        prey_6: random1
        prey_7: random1
        prey_8: random1
        prey_9: random1
        prey_10: random1
        prey_11: random1
        prey_12: random1
        prey_13: random1
        prey_14: random1
        prey_15: random1
        prey_16: random1
        prey_17: random1
        prey_18: random1
        prey_19: random1
        prey_20: random1
        prey_21: random1
        prey_22: random1
        prey_23: random1
        prey_24: random1
        prey_25: random1
        prey_26: random1
        prey_27: random1
        prey_28: random1
        prey_29: random1
        prey_30: random1
        prey_31: random1
        prey_32: random1
        prey_33: random1
        prey_34: random1
        prey_35: random1
        prey_36: random1
        prey_37: random1
        prey_38: random1
        prey_39: random1
        prey_40: random1
        prey_41: random1
        prey_42: random1
        prey_43: random1
        prey_44: random1
        prey_45: random1
        prey_46: random1
        prey_47: random1
        prey_48: random1
        prey_49: random1
    opp_obs_queue_len: 1
    channel_first: true

critic:
  type: "QMixer"
  model:
    model_type: QMixModel
    state_shape: 256
    input_dim: 25
    qmix_hidden_dim: 32
    hypernet_layers: 2
    hyper_hidden_dim: 32
  feature_extractor:
    model_type: "Custom"
    layers:
      - type: Conv2d
        in_channels: 29
        out_channels: 32
        kernel_size: 9
        stride: 4
        padding: 4
      - type: BatchNorm2d
        num_features: 32
      - type: LeakyReLU
      - type: Conv2d
        in_channels: 32
        out_channels: 64
        kernel_size: 4
        stride: 2
        padding: 1
      - type: LeakyReLU
      - type: BatchNorm2d
        num_features: 64
      - type: AdaptiveAvgPool2d
        output_size: [2,2]
      - type: Flatten
  optimizer:
    type: "Adam"
    lr: 0.0002
    weight_decay: 0.00005

rollout:
  manager_type: "multi-process"
  worker_type: "multi-process"
  n_workers: 1
  n_episodes: 1
  n_eval_episodes: 1
  traj_len: 1
  episode_limit: 2
  device: "cpu"

replay_buffer:
  type: "Prioritized"
  capacity: 2
  traj_len: 1
  priority_attr: "all_agents_sum_rewards"

analyzer:
  type: "default"

trainer:
  type: "GraphQMIX"
  gamma: 0.95
  eval_epsilon: 0.01
  workdir: "./test/results/replace_by_tempfile"
  train_device: "cpu"
  n_workers: 0

  epsilon_scheduler:
    type: "linear"
    start_value: 1.0
    end_value: 0.1
    decay_steps: 10

  sample_ratio_scheduler:
    type: "linear"
    start_value: 16
    end_value: 16
    decay_steps: 10

  train_args:
    epochs: 1
    target_first_metric: 1000
    rollback_interval: 4
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
            n_episodes = 4
            self.trainer.collect_experience(0.9)
            self.assertNotEqual(len(self.trainer.replaybuffer.buffer), 0)

    def test_learn(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self._create_trainer(temp_dir)
            origin_critic_params = deepcopy(self.trainer.target_critic.state_dict())
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=1)
            self.trainer._update_target_after_batch()
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
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self._create_trainer(temp_dir)
            result = self.trainer.evaluate()
            best_metrics = self.trainer.train(epochs=2, target_first_metric=5)

    def test_distributed_data_parallel(self):
        """Test DistributedDataParallel training with proper DDP initialization."""
        config = deepcopy(self.config)
        config["trainer"]["train_device"] = ["cuda:0"]

        with tempfile.TemporaryDirectory() as temp_dir:
            config["trainer"]["workdir"] = temp_dir
            trainer_config = TrainerConfig(config)
            self.trainer = trainer_config.create_trainer()
            origin_critic_params = deepcopy(self.trainer.target_critic.state_dict())
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=1)
            self.trainer._update_target_after_batch()
            critic_params = self.trainer.target_critic.state_dict()

            for w1, w2 in zip(critic_params.values(), origin_critic_params.values()):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))

    def test_torch_compile(self):
        config = deepcopy(self.config)
        config["trainer"]["compile_models"] = True

        with tempfile.TemporaryDirectory() as temp_dir:
            config["trainer"]["workdir"] = temp_dir
            trainer_config = TrainerConfig(config)
            self.trainer = trainer_config.create_trainer()
            origin_critic_params = deepcopy(self.trainer.target_critic.state_dict())
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=1)
            self.trainer._update_target_after_batch()
            critic_params = self.trainer.target_critic.state_dict()

            for w1, w2 in zip(critic_params.values(), origin_critic_params.values()):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))


if __name__ == "__main__":
    unittest.main()
