import unittest
import yaml
import tempfile
import torch
from copy import deepcopy
from marlite.trainer import TrainerConfig, Trainer


class TestTrainerConfig(unittest.TestCase):
    def setUp(self):
        yaml_str = """
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
      encoder:
        model_type: "RNN"
        input_shape: 18
        rnn_hidden_dim: 32
        rnn_layers: 1
        output_shape: 32
      decoder:
        model_type: "Custom"
        layers:
        - type: "Linear"
          in_features: 32
          out_features: 5
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
  episode_limit: 3
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
"""
        self.config = yaml.safe_load(yaml_str)

        if torch.cuda.is_available():
            self.config["trainer"]["train_device"] = "cuda"
            self.config["rollout"]["device"] = "cuda"

    def _create_trainer_config(self, temp_dir):
        config = deepcopy(self.config)
        config['trainer']['workdir'] = temp_dir
        return TrainerConfig(config)

    def test_create_learner(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer_config = self._create_trainer_config(temp_dir)
            ret = trainer_config.create_trainer()
            self.assertTrue(isinstance(ret, Trainer))

    def test_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer_config = self._create_trainer_config(temp_dir)
            best_metrics = trainer_config.run()


@unittest.skip("KAZ env incompatible with pygame 2.6.1 SRCALPHA surfaces on headless systems")
class TestTrainerConfigWithKAZConfig(unittest.TestCase):
    def setUp(self):
        yaml_str = """
agent_group:
  type: "QMIX"
  agent_list:
    archer_0: model1
    archer_1: model1
    knight_0: model2
    knight_1: model2
  models:
    model1:
      feature_extractor:
        model_type: "Custom"
        layers:
        - type: Flatten
        - type: Linear
          in_features: 135
          out_features: 16
        - type: LeakyReLU
      encoder:
        model_type: "CustomConv1D"
        layers:
        - type: Conv1d
          in_channels: 16
          out_channels: 16
          kernel_size: 5
          stride: 1
          padding: 0
        - type: Flatten
        - type: LeakyReLU
        - type: Linear
          in_features: 16
          out_features: 16
      decoder:
        model_type: "Custom"
        layers:
        - type: "Linear"
          in_features: 16
          out_features: 6
    model2:
      feature_extractor:
        model_type: "Custom"
        layers:
        - type: Flatten
        - type: Linear
          in_features: 135
          out_features: 64
        - type: LeakyReLU
      encoder:
        model_type: "CustomConv1D"
        layers:
        - type: Conv1d
          in_channels: 64
          out_channels: 16
          kernel_size: 3
          stride: 2
          padding: 0
        - type: Flatten
        - type: LeakyReLU
        - type: Linear
          in_features: 32
          out_features: 32
      decoder:
        model_type: "Custom"
        layers:
        - type: "Linear"
          in_features: 32
          out_features: 6
  optimizer:
    type: "Adam"
    lr: 0.0005
    weight_decay: 0.0001

environment:
  module_name: "pettingzoo.butterfly"
  env_name: "knights_archers_zombies_v10"
  env_params:
    sequence_space: false

critic:
  type: "QMixer"
  model:
    model_type: QMixModel
    state_shape: 104
    input_dim: 4
    qmix_hidden_dim: 32
  feature_extractor:
    model_type: "Flatten"
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
  traj_len: 5
  episode_limit: 3
  device: "cpu"

replay_buffer:
  type: "Prioritized"
  capacity: 5
  traj_len: 5
  priority_attr: "all_agents_sum_rewards"

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
"""
        self.config = yaml.safe_load(yaml_str)

        if torch.cuda.is_available():
            self.config["trainer"]["train_device"] = "cuda"
            self.config["rollout"]["device"] = "cuda"

    def _create_trainer_config(self, temp_dir):
        config = deepcopy(self.config)
        config['trainer']['workdir'] = temp_dir
        return TrainerConfig(config)

    def test_create_learner(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer_config = self._create_trainer_config(temp_dir)
            ret = trainer_config.create_trainer()
            self.assertTrue(isinstance(ret, Trainer))

    def test_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer_config = self._create_trainer_config(temp_dir)
            best_metrics = trainer_config.run()


class TestTrainerConfigWithMAgentPredator(unittest.TestCase):
    def setUp(self):
        yaml_str = """
agent_group:
  type: "QMIX"
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
          in_channels: 5
          out_channels: 16
          kernel_size: 3
          stride: 1
          padding: 1
        - type: ReLU
        - type: MaxPool2d
          kernel_size: 2
          stride: 2
        - type: Conv2d
          in_channels: 16
          out_channels: 32
          kernel_size: 3
          stride: 1
          padding: 1
        - type: ReLU
        - type: MaxPool2d
          kernel_size: 2
          stride: 2
        - type: Conv2d
          in_channels: 32
          out_channels: 64
          kernel_size: 3
          stride: 1
          padding: 1
        - type: ReLU
        - type: AdaptiveAvgPool2d
          output_size: [1, 1]
        - type: Flatten
        - type: Linear
          in_features: 64
          out_features: 64
      encoder:
        model_type: "RNN"
        input_shape: 64
        rnn_hidden_dim: 32
        rnn_layers: 1
        output_shape: 32
      decoder:
        model_type: "Custom"
        layers:
        - type: "Linear"
          in_features: 32
          out_features: 13
  optimizer:
    type: "Adam"
    lr: 0.0005
    weight_decay: 0.0001

environment:
  module_name: "magent2.environments"
  env_name: "adversarial_pursuit_v4"
  env_params:
    tag_penalty: -0.01
    extra_features: false
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
  feature_extractor:
    model_type: "Custom"
    layers:
      - type: Conv2d
        in_channels: 5
        out_channels: 16
        kernel_size: 3
        stride: 1
        padding: 1
      - type: ReLU
      - type: MaxPool2d
        kernel_size: 2
        stride: 2
        padding: 0
      - type: Conv2d
        in_channels: 16
        out_channels: 32
        kernel_size: 3
        stride: 1
        padding: 1
      - type: ReLU
      - type: MaxPool2d
        kernel_size: 2
        stride: 2
        padding: 0
      - type: Conv2d
        in_channels: 32
        out_channels: 64
        kernel_size: 3
        stride: 1
        padding: 1
      - type: ReLU
      - type: MaxPool2d
        kernel_size: 2
        stride: 2
        padding: 0
      - type: Flatten
      - type: Linear
        in_features: 1600
        out_features: 256
      - type: ReLU
      - type: Dropout
        p: 0.5
      - type: Linear
        in_features: 256
        out_features: 256
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
  traj_len: 5
  episode_limit: 3
  device: "cpu"

replay_buffer:
  type: "Normal"
  capacity: 5
  traj_len: 5

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
"""
        self.config = yaml.safe_load(yaml_str)
        if torch.cuda.is_available():
            self.config['trainer']['train_device'] = 'cuda'
            self.config['rollout']['device'] = 'cpu'
        if torch.cuda.is_available():
            self.config["trainer"]["train_device"] = "cuda"
            self.config["rollout"]["device"] = "cuda"

    def _create_trainer_config(self, temp_dir):
        config = deepcopy(self.config)
        config['trainer']['workdir'] = temp_dir
        return TrainerConfig(config)

    def test_create_learner(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer_config = self._create_trainer_config(temp_dir)
            ret = trainer_config.create_trainer()
            self.assertTrue(isinstance(ret, Trainer))

    def test_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer_config = self._create_trainer_config(temp_dir)
            best_metrics = trainer_config.run()


class TestTrainerConfigWithMAgentPrey(unittest.TestCase):
    def setUp(self):
        yaml_str = """
agent_group:
  type: "QMIX"
  agent_list:
    prey_0: model1
    prey_1: model1
    prey_2: model1
    prey_3: model1
    prey_4: model1
    prey_5: model1
    prey_6: model1
    prey_7: model1
    prey_8: model1
    prey_9: model1
    prey_10: model1
    prey_11: model1
    prey_12: model1
    prey_13: model1
    prey_14: model1
    prey_15: model1
    prey_16: model1
    prey_17: model1
    prey_18: model1
    prey_19: model1
    prey_20: model1
    prey_21: model1
    prey_22: model1
    prey_23: model1
    prey_24: model1
    prey_25: model1
    prey_26: model1
    prey_27: model1
    prey_28: model1
    prey_29: model1
    prey_30: model1
    prey_31: model1
    prey_32: model1
    prey_33: model1
    prey_34: model1
    prey_35: model1
    prey_36: model1
    prey_37: model1
    prey_38: model1
    prey_39: model1
    prey_40: model1
    prey_41: model1
    prey_42: model1
    prey_43: model1
    prey_44: model1
    prey_45: model1
    prey_46: model1
    prey_47: model1
    prey_48: model1
    prey_49: model1
  models:
    model1:
      feature_extractor:
        model_type: "Custom"
        layers:
          - type: Conv2d
            in_channels: 5
            out_channels: 16
            kernel_size: 3
            stride: 1
            padding: 1
          - type: ReLU
          - type: MaxPool2d
            kernel_size: 2
            stride: 2
            padding: 0
          - type: Conv2d
            in_channels: 16
            out_channels: 32
            kernel_size: 3
            stride: 1
            padding: 1
          - type: ReLU
          - type: MaxPool2d
            kernel_size: 2
            stride: 2
            padding: 0
          - type: Flatten
          - type: Linear
            in_features: 128
            out_features: 64
            bias: True
      encoder:
        model_type: "RNN"
        input_shape: 64
        rnn_hidden_dim: 32
        rnn_layers: 1
        output_shape: 32
      decoder:
        model_type: "Custom"
        layers:
        - type: "Linear"
          in_features: 32
          out_features: 9

  optimizer:
    type: "Adam"
    lr: 0.0005
    weight_decay: 0.0001

environment:
  module_name: "magent2.environments"
  env_name: "adversarial_pursuit_v4"
  env_params:
    tag_penalty: -0.01
    extra_features: false
  wrapper:
    type: adversarial_pursuit_prey
    opponent_agent_group:
      type: "Random"
      agent_list:
        predator_0: random1
        predator_1: random1
        predator_2: random1
        predator_3: random1
        predator_4: random1
        predator_5: random1
        predator_6: random1
        predator_7: random1
        predator_8: random1
        predator_9: random1
        predator_10: random1
        predator_11: random1
        predator_12: random1
        predator_13: random1
        predator_14: random1
        predator_15: random1
        predator_16: random1
        predator_17: random1
        predator_18: random1
        predator_19: random1
        predator_20: random1
        predator_21: random1
        predator_22: random1
        predator_23: random1
        predator_24: random1
    opp_obs_queue_len: 5
    channel_first: true

critic:
  type: "QMixer"
  model:
    model_type: QMixModel
    state_shape: 256
    input_dim: 50
    qmix_hidden_dim: 32
  feature_extractor:
    model_type: "Custom"
    layers:
      - type: Conv2d
        in_channels: 5
        out_channels: 16
        kernel_size: 3
        stride: 1
        padding: 1
      - type: ReLU
      - type: MaxPool2d
        kernel_size: 2
        stride: 2
        padding: 0
      - type: Conv2d
        in_channels: 16
        out_channels: 32
        kernel_size: 3
        stride: 1
        padding: 1
      - type: ReLU
      - type: MaxPool2d
        kernel_size: 2
        stride: 2
        padding: 0
      - type: Conv2d
        in_channels: 32
        out_channels: 64
        kernel_size: 3
        stride: 1
        padding: 1
      - type: ReLU
      - type: MaxPool2d
        kernel_size: 2
        stride: 2
        padding: 0
      - type: Flatten
      - type: Linear
        in_features: 1600
        out_features: 256
      - type: ReLU
      - type: Dropout
        p: 0.5
      - type: Linear
        in_features: 256
        out_features: 256
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
  traj_len: 5
  episode_limit: 3
  device: "cpu"

replay_buffer:
  type: "Normal"
  capacity: 5
  traj_len: 5

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
"""
        self.config = yaml.safe_load(yaml_str)
        if torch.cuda.is_available():
            self.config['trainer']['train_device'] = 'cuda'
            self.config['rollout']['device'] = 'cpu'
        if torch.cuda.is_available():
            self.config["trainer"]["train_device"] = "cuda"
            self.config["rollout"]["device"] = "cuda"

    def _create_trainer_config(self, temp_dir):
        config = deepcopy(self.config)
        config['trainer']['workdir'] = temp_dir
        return TrainerConfig(config)

    def test_create_learner(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer_config = self._create_trainer_config(temp_dir)
            ret = trainer_config.create_trainer()
            self.assertTrue(isinstance(ret, Trainer))

    def test_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer_config = self._create_trainer_config(temp_dir)
            best_metrics = trainer_config.run()


class TestTrainerConfigWithMAgentBattlefield(unittest.TestCase):
    def setUp(self):
        yaml_str = """
agent_group:
  type: "QMIX"
  agent_list:
    prey_0: model1
    prey_1: model1
    prey_2: model1
    prey_3: model1
    prey_4: model1
    prey_5: model1
    prey_6: model1
    prey_7: model1
    prey_8: model1
    prey_9: model1
    prey_10: model1
    prey_11: model1
    prey_12: model1
    prey_13: model1
    prey_14: model1
    prey_15: model1
    prey_16: model1
    prey_17: model1
    prey_18: model1
    prey_19: model1
    prey_20: model1
    prey_21: model1
    prey_22: model1
    prey_23: model1
    prey_24: model1
    prey_25: model1
    prey_26: model1
    prey_27: model1
    prey_28: model1
    prey_29: model1
    prey_30: model1
    prey_31: model1
    prey_32: model1
    prey_33: model1
    prey_34: model1
    prey_35: model1
    prey_36: model1
    prey_37: model1
    prey_38: model1
    prey_39: model1
    prey_40: model1
    prey_41: model1
    prey_42: model1
    prey_43: model1
    prey_44: model1
    prey_45: model1
    prey_46: model1
    prey_47: model1
    prey_48: model1
    prey_49: model1
  models:
    model1:
      feature_extractor:
        model_type: "Custom"
        layers:
          - type: Conv2d
            in_channels: 5
            out_channels: 16
            kernel_size: 3
            stride: 1
            padding: 1
          - type: ReLU
          - type: MaxPool2d
            kernel_size: 2
            stride: 2
            padding: 0
          - type: Conv2d
            in_channels: 16
            out_channels: 32
            kernel_size: 3
            stride: 1
            padding: 1
          - type: ReLU
          - type: MaxPool2d
            kernel_size: 2
            stride: 2
            padding: 0
          - type: Flatten
          - type: Linear
            in_features: 128
            out_features: 64
            bias: True
      encoder:
        model_type: "RNN"
        input_shape: 64
        rnn_hidden_dim: 32
        rnn_layers: 1
        output_shape: 32
      decoder:
        model_type: "Custom"
        layers:
        - type: "Linear"
          in_features: 32
          out_features: 9

  optimizer:
    type: "Adam"
    lr: 0.0005
    weight_decay: 0.0001

environment:
  module_name: "magent2.environments"
  env_name: "adversarial_pursuit_v4"
  env_params:
    tag_penalty: -0.01
    extra_features: false
  wrapper:
    type: adversarial_pursuit_prey
    opponent_agent_group:
      type: "Random"
      agent_list:
        predator_0: random1
        predator_1: random1
        predator_2: random1
        predator_3: random1
        predator_4: random1
        predator_5: random1
        predator_6: random1
        predator_7: random1
        predator_8: random1
        predator_9: random1
        predator_10: random1
        predator_11: random1
        predator_12: random1
        predator_13: random1
        predator_14: random1
        predator_15: random1
        predator_16: random1
        predator_17: random1
        predator_18: random1
        predator_19: random1
        predator_20: random1
        predator_21: random1
        predator_22: random1
        predator_23: random1
        predator_24: random1
    opp_obs_queue_len: 5
    channel_first: true

critic:
  type: "QMixer"
  model:
    model_type: QMixModel
    state_shape: 256
    input_dim: 50
    qmix_hidden_dim: 32
  feature_extractor:
    model_type: "Custom"
    layers:
      - type: Conv2d
        in_channels: 5
        out_channels: 16
        kernel_size: 3
        stride: 1
        padding: 1
      - type: ReLU
      - type: MaxPool2d
        kernel_size: 2
        stride: 2
        padding: 0
      - type: Conv2d
        in_channels: 16
        out_channels: 32
        kernel_size: 3
        stride: 1
        padding: 1
      - type: ReLU
      - type: MaxPool2d
        kernel_size: 2
        stride: 2
        padding: 0
      - type: Conv2d
        in_channels: 32
        out_channels: 64
        kernel_size: 3
        stride: 1
        padding: 1
      - type: ReLU
      - type: MaxPool2d
        kernel_size: 2
        stride: 2
        padding: 0
      - type: Flatten
      - type: Linear
        in_features: 1600
        out_features: 256
      - type: ReLU
      - type: Dropout
        p: 0.5
      - type: Linear
        in_features: 256
        out_features: 256
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
  traj_len: 5
  episode_limit: 3
  device: "cpu"

replay_buffer:
  type: "Normal"
  capacity: 5
  traj_len: 5

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
"""
        self.config = yaml.safe_load(yaml_str)
        if torch.cuda.is_available():
            self.config['trainer']['train_device'] = 'cuda'
            self.config['rollout']['device'] = 'cpu'
        if torch.cuda.is_available():
            self.config["trainer"]["train_device"] = "cuda"
            self.config["rollout"]["device"] = "cuda"

    def _create_trainer_config(self, temp_dir):
        config = deepcopy(self.config)
        config['trainer']['workdir'] = temp_dir
        return TrainerConfig(config)

    def test_create_learner(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer_config = self._create_trainer_config(temp_dir)
            ret = trainer_config.create_trainer()
            self.assertTrue(isinstance(ret, Trainer))

    def test_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer_config = self._create_trainer_config(temp_dir)
            best_metrics = trainer_config.run()
