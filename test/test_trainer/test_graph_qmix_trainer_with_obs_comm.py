import unittest
import yaml
from copy import deepcopy
import tempfile
import torch

from marlite.trainer import TrainerConfig


class TestObsGNNCommQMIXTrainer(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load("""
agent_group:
  type: "ObsGNNComm"
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
          in_features: 96
          out_features: 13
  graph_builder:
    type: "PartialMAgent"
    binary_agent_id_dim: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    agent_presence_dim: [1, 3]
    comm_distance: 20
    distance_metric: "cityblock"
    n_workers: 20
    n_subgraphs: 5
    valid_node_list: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    target_node_list: [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
    update_interval: 5
  graph_model:
    model_type: "GAT"
    input_dim: 64
    hidden_dim: 32
    output_dim: 32
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
            self.trainer.update_target_model_params()
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
            self.trainer.update_target_model_params()
            critic_params = self.trainer.target_critic.state_dict()

            for w1, w2 in zip(critic_params.values(), origin_critic_params.values()):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))


class TestPartialObsGNNCommQMIXTrainer(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load("""
agent_group:
  type: "ObsGNNComm"
  agent_list:
    red_0: model_0
    red_1: model_0
    red_2: model_0
    red_3: model_0
    red_4: model_0
    red_5: model_0
    red_6: model_0
    red_7: model_0
    red_8: model_0
    red_9: model_0
    red_10: model_0
    red_11: model_0
    red_12: model_0
    red_13: model_0
    red_14: model_0
    red_15: model_0
    red_16: model_0
    red_17: model_0
    red_18: model_0
    red_19: model_0
    red_20: model_0
    red_21: model_0
    red_22: model_0
    red_23: model_0
    red_24: model_0
    red_25: model_0
    red_26: model_0
    red_27: model_0
    red_28: model_0
    red_29: model_0
    red_30: model_0
    red_31: model_0
    red_32: model_0
    red_33: model_0
    red_34: model_0
    red_35: model_0
  models:
    model_0:
      feature_extractor:
        model_type: "Custom"
        layers:
        - type: Conv2d
          in_channels: 37
          out_channels: 16
          kernel_size: 1
          stride: 1
          padding: 0
        - type: BatchNorm2d
          num_features: 16
        - type: GELU
        - type: Conv2d
          in_channels: 16
          out_channels: 4
          kernel_size: 1
          stride: 1
          padding: 0
        - type: BatchNorm2d
          num_features: 4
        - type: GELU
        - type: Flatten
        - type: Linear
          in_features: 676
          out_features: 64
      encoder:
        model_type: "SimpleResAttSeqEnc"
        input_dim: 64
        embed_dim: 32
        output_dim: 32
        num_heads: 2
        max_seq_len: 8
        dropout: 0.0
      decoder:
        model_type: "Custom"
        layers:
        - type: GELU
        - type: Linear
          in_features: 64
          out_features: 21
  graph_builder:
    type: "PartialMAgent"
    binary_agent_id_dim: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    agent_presence_dim: [1, 3]
    comm_distance: 8
    distance_metric: "cityblock"
    n_workers: 32
    n_subgraphs: 8
    valid_node_list: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
    target_node_list: [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
    update_interval: 5
  graph_model:
    model_type: "GAT"
    input_dim: 64
    hidden_dim: 32
    output_dim: 32
    head_conv1: 2
    head_conv2: 2
    dropout: 0.5
    activation: ELU
  optimizer:
    type: "Adam"
    lr: 0.001
    weight_decay: 0.00001
  lr_scheduler:
    type: "ReduceLROnPlateau"
    mode: "max"
    factor: 0.5
    patience: 4
    min_lr: 0.00001

environment:
  module_name: "magent2.environments"
  env_name: "battle_v4"
  env_params:
    map_size: 32
    step_reward: -0.00
    dead_penalty: -1.0
    attack_penalty: -0.05
    attack_opponent_reward: 1.0
    extra_features: true
  wrapper:
    type: battle
    opp_obs_queue_len: 1
    channel_first: true
    vector_state: false
    opponent_agent_group:
      type: "MAgentBattle"
      agent_list:
        blue_0: policy
        blue_1: policy
        blue_2: policy
        blue_3: policy
        blue_4: policy
        blue_5: policy
        blue_6: policy
        blue_7: policy
        blue_8: policy
        blue_9: policy
        blue_10: policy
        blue_11: policy
        blue_12: policy
        blue_13: policy
        blue_14: policy
        blue_15: policy
        blue_16: policy
        blue_17: policy
        blue_18: policy
        blue_19: policy
        blue_20: policy
        blue_21: policy
        blue_22: policy
        blue_23: policy
        blue_24: policy
        blue_25: policy
        blue_26: policy
        blue_27: policy
        blue_28: policy
        blue_29: policy
        blue_30: policy
        blue_31: policy
        blue_32: policy
        blue_33: policy
        blue_34: policy
        blue_35: policy

critic:
  type: "QMixer"
  model:
    model_type: QMixModel
    state_shape: 256
    input_dim: 36
    qmix_hidden_dim: 32
    hypernet_layers: 2
    hyper_hidden_dim: 32
  feature_extractor:
    model_type: "Custom"
    layers:
      - type: Conv2d
        in_channels: 37
        out_channels: 32
        kernel_size: 9
        stride: 4
        padding: 4
      - type: BatchNorm2d
        num_features: 32
      - type: GELU
      - type: Conv2d
        in_channels: 32
        out_channels: 64
        kernel_size: 4
        stride: 2
        padding: 1
      - type: GELU
      - type: BatchNorm2d
        num_features: 64
      - type: AdaptiveAvgPool2d
        output_size: [2,2]
      - type: Flatten
  optimizer:
    type: "Adam"
    lr: 0.001
    weight_decay: 0.00001
  lr_scheduler:
    type: "ReduceLROnPlateau"
    mode: "max"
    factor: 0.5
    patience: 4
    min_lr: 0.00001

rollout:
  manager_type: "multi-process"
  worker_type: "multi-process"
  n_workers: 1
  n_episodes: 1
  n_eval_episodes: 1
  traj_len: 8
  episode_limit: 2
  victory_checker: "default"
  device: "cpu"

replay_buffer:
  type: "Prioritized"
  capacity: 2
  traj_len: 8
  priority_attr: "all_agents_sum_rewards"

analyzer:
  type: "default"

trainer:
  type: "GraphQMIX"
  gamma: 0.95
  eval_epsilon: 0.005
  eval_episodes_to_replay_ratio: 0.25
  workdir: "./test/results/replace_by_tempfile"
  train_device: "cpu"
  compile_models: false
  n_workers: 0

  epsilon_scheduler:
    type: "linear"
    start_value: 1.0
    end_value: 0.05
    decay_steps: 10

  sample_ratio_scheduler:
    type: "logarithmic"
    start_value: 16
    end_value: 16
    decay_steps: 10

  train_args:
    epochs: 1
    target_first_metric: 10000000
    rollback_interval: 4
    update_target_interval: 2
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
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self._create_trainer(temp_dir)
            result = self.trainer.evaluate()
            best_metrics = self.trainer.train(epochs=2, target_first_metric=5)


if __name__ == "__main__":
    unittest.main()
