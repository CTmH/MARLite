import unittest
import os
import yaml
from copy import deepcopy
import tempfile
import torch

from marlite.trainer import TrainerConfig


class TestMsgAggrQMIXTrainer(unittest.TestCase):
    yaml_config = """
agent_group:
  type: "MsgAggr"
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
        - type: ReLU
        - type: Conv2d
          in_channels: 8
          out_channels: 4
          kernel_size: 3
          stride: 1
          padding: 0
        - type: BatchNorm2d
          num_features: 4
        - type: Flatten
        - type: Linear
          in_features: 256
          out_features: 32
      encoder:
        model_type: "CustomConv1D"
        layers:
        - type: Conv1d
          in_channels: 32
          out_channels: 32
          kernel_size: 5
          stride: 1
          padding: 0
        - type: Flatten
        - type: Linear
          in_features: 32
          out_features: 32
        - type: ReLU
      decoder:
        model_type: "Custom"
        layers:
        - type: Linear
          in_features: 64
          out_features: 13
  aggr_model_config:
    model_type: "Custom"
    layers:
        - type: SelfAttention
          embed_dim: 32
          num_heads: 1
          batch_first: true
        - type: Permute
          dims: [0, 2, 1]
        - type: AdaptiveAvgPool1d
          output_size: 1
        - type: Flatten
        - type: Linear
          in_features: 32
          out_features: 32
        - type: ReLU
  optimizer:
    type: "Adam"
    lr: 0.0002
    weight_decay: 0.00005

environment:
  module_name: "magent2.environments"
  env_name: "adversarial_pursuit_v4"
  env_params:
    tag_penalty: -0.01
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
    state_shape: 32
    input_dim: 25
    qmix_hidden_dim: 32
    hypernet_layers: 1
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
      - type: Linear
        in_features: 256
        out_features: 32
      - type: ReLU
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
  traj_len: 5
  episode_limit: 2
  device: "cpu"

replay_buffer:
  type: "Prioritized"
  capacity: 2
  traj_len: 5
  priority_attr: "all_agents_sum_rewards"

analyzer:
  type: "default"

trainer:
  type: "MsgAggr"
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
"""

    def setUp(self):
        self.config = yaml.safe_load(self.yaml_config)

        if torch.cuda.is_available():
            self.config["trainer"]["train_device"] = "cuda"
            self.config["rollout"]["device"] = "cuda"

    def _create_trainer(self, temp_dir, config=None):
        if config is None:
            config = self.config
        config['trainer']['workdir'] = temp_dir
        return TrainerConfig(config).create_trainer()

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
            origin_agent_group_params = deepcopy(
                self.trainer.target_agent_group.state_dict()
            )
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=1)
            self.trainer._update_target_after_batch()
            critic_params = self.trainer.target_critic.state_dict()
            agent_group_params = self.trainer.target_agent_group.state_dict()

            for (w_name, w1), w2 in zip(
                critic_params.items(), origin_critic_params.values()
            ):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))

            for name in agent_group_params:
                if name in origin_agent_group_params:
                    if isinstance(agent_group_params[name], torch.Tensor):
                        if agent_group_params[name].requires_grad:
                            self.assertFalse(
                                torch.equal(
                                    agent_group_params[name],
                                    origin_agent_group_params[name],
                                )
                            )

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


class TestMsgAggrSMACQMIXTrainer(unittest.TestCase):
    yaml_config = """
agent_group:
  type: "MsgAggr"
  agent_list:
    stalker_0: model_0
    stalker_1: model_0
    zealot_0: model_1
    zealot_1: model_1
    zealot_2: model_1
  models:
    model_0:
      feature_extractor:
        model_type: "Custom"
        layers:
        - type: Linear
          in_features: 144
          out_features: 32
      encoder:
        model_type: "SimpleResAttSeqEnc"
        input_dim: 32
        embed_dim: 32
        output_dim: 32
        num_heads: 2
        max_seq_len: 5
        dropout: 0.25
      decoder:
        model_type: "Custom"
        layers:
        - type: Linear
          in_features: 64
          out_features: 11
    model_1:
      feature_extractor:
        model_type: "Custom"
        layers:
        - type: Linear
          in_features: 144
          out_features: 32
      encoder:
        model_type: "CustomConv1D"
        layers:
        - type: Conv1d
          in_channels: 32
          out_channels: 32
          kernel_size: 5
          stride: 1
          padding: 0
        - type: Flatten
        - type: Linear
          in_features: 32
          out_features: 32
        - type: ReLU
      decoder:
        model_type: "Custom"
        layers:
        - type: Linear
          in_features: 64
          out_features: 11
  aggr_model_config:
    model_type: "Custom"
    layers:
        - type: SelfAttention
          embed_dim: 32
          num_heads: 1
          batch_first: true
        - type: Permute
          dims: [0, 2, 1]
        - type: AdaptiveAvgPool1d
          output_size: 1
        - type: Flatten
        - type: Linear
          in_features: 32
          out_features: 32
        - type: ReLU
  optimizer:
    type: "Adam"
    lr: 0.0005
    weight_decay: 0.00005
  lr_scheduler:
    type: "ReduceLROnPlateau"
    mode: "max"
    patience: 3

environment:
  module_name: "smac_pettingzoo"
  env_name: "smacv1_pettingzoo_v1"
  env_params:
    map_name: "2s3z"
  wrapper:
    type: smac

critic:
  type: "QMixer"
  model:
    model_type: QMixModel
    state_shape: 32
    input_dim: 5
    qmix_hidden_dim: 16
    hypernet_layers: 1
    hyper_hidden_dim: 16
  feature_extractor:
    model_type: "ResAttMaskedStateEnc"
    input_dim: 173
    embed_dim: 32
    num_heads: 4
    max_seq_len: 5
    dropout: 0.25
  optimizer:
    type: "Adam"
    lr: 0.0005
    weight_decay: 0.00005
  lr_scheduler:
    type: "ReduceLROnPlateau"
    mode: "max"
    patience: 3

rollout:
  manager_type: "persistent-env"
  worker_type: "persistent-env"
  n_workers: 1
  n_episodes: 1
  n_eval_episodes: 1
  traj_len: 5
  episode_limit: 2
  device: "cpu"
  victory_checker: smac

replay_buffer:
  type: "Prioritized"
  capacity: 2
  traj_len: 5
  priority_attr: "all_agents_sum_rewards"

analyzer:
  type: "default"

trainer:
  type: "MsgAggr"
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
"""

    def setUp(self):
        self.config = yaml.safe_load(self.yaml_config)

        if torch.cuda.is_available():
            self.config["trainer"]["train_device"] = "cuda"
            self.config["rollout"]["device"] = "cuda"

    def _create_trainer(self, temp_dir, config=None):
        if config is None:
            config = self.config
        config['trainer']['workdir'] = temp_dir
        return TrainerConfig(config).create_trainer()

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
            origin_agent_group_params = deepcopy(
                self.trainer.target_agent_group.state_dict()
            )
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=1)
            self.trainer._update_target_after_batch()
            critic_params = self.trainer.target_critic.state_dict()
            agent_group_params = self.trainer.target_agent_group.state_dict()

            for (w_name, w1), w2 in zip(
                critic_params.items(), origin_critic_params.values()
            ):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))

            for name in agent_group_params:
                if name in origin_agent_group_params:
                    if isinstance(agent_group_params[name], torch.Tensor):
                        if agent_group_params[name].requires_grad:
                            self.assertFalse(
                                torch.equal(
                                    agent_group_params[name],
                                    origin_agent_group_params[name],
                                )
                            )

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
        gpu_count = torch.cuda.device_count()
        if gpu_count < 2:
            self.skipTest(f"Need at least 2 GPUs for multi-GPU test, found {gpu_count}")
        config = deepcopy(self.config)
        config['trainer']['train_device'] = [f"cuda:{i}" for i in range(gpu_count)]
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self._create_trainer(temp_dir, config=config)
            origin_critic_params = deepcopy(self.trainer.eval_critic.state_dict())
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=1)
            self.trainer._update_target_after_batch()
            critic_params = self.trainer.eval_critic.state_dict()

            for w1, w2 in zip(critic_params.values(), origin_critic_params.values()):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))

    def test_torch_compile(self):
        config = deepcopy(self.config)
        config['trainer']['compile_models'] = True
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self._create_trainer(temp_dir, config=config)
            origin_critic_params = deepcopy(self.trainer.target_critic.state_dict())
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=1)
            self.trainer._update_target_after_batch()
            critic_params = self.trainer.target_critic.state_dict()

            for w1, w2 in zip(critic_params.values(), origin_critic_params.values()):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))


class TestSeqMsgAggrSMACQMIXTrainer(unittest.TestCase):
    yaml_config = """
agent_group:
  type: "SeqMsgAggr"
  agent_list:
    stalker_0: model_0
    stalker_1: model_0
    zealot_0: model_1
    zealot_1: model_1
    zealot_2: model_1
  models:
    model_0:
      feature_extractor:
        model_type: "Custom"
        layers:
        - type: Linear
          in_features: 144
          out_features: 32
      encoder:
        model_type: "SimpleResAttSeqEnc"
        input_dim: 32
        embed_dim: 32
        output_dim: 32
        num_heads: 2
        max_seq_len: 5
        dropout: 0.25
      decoder:
        model_type: "Custom"
        layers:
        - type: Linear
          in_features: 64
          out_features: 11
    model_1:
      feature_extractor:
        model_type: "Custom"
        layers:
        - type: Linear
          in_features: 144
          out_features: 32
      encoder:
        model_type: "CustomConv1D"
        layers:
        - type: Conv1d
          in_channels: 32
          out_channels: 32
          kernel_size: 5
          stride: 1
          padding: 0
        - type: Flatten
        - type: Linear
          in_features: 32
          out_features: 32
        - type: ReLU
      decoder:
        model_type: "Custom"
        layers:
        - type: Linear
          in_features: 64
          out_features: 11
  aggr_model_config:
    model_type: "Custom"
    layers:
        - type: SelfAttention
          embed_dim: 32
          num_heads: 1
          batch_first: true
        - type: Permute
          dims: [0, 2, 1]
        - type: AdaptiveAvgPool1d
          output_size: 1
        - type: Flatten
        - type: Linear
          in_features: 32
          out_features: 32
        - type: ReLU
  optimizer:
    type: "Adam"
    lr: 0.0005
    weight_decay: 0.00005
  lr_scheduler:
    type: "ReduceLROnPlateau"
    mode: "max"
    patience: 3

environment:
  module_name: "smac_pettingzoo"
  env_name: "smacv1_pettingzoo_v1"
  env_params:
    map_name: "2s3z"
  wrapper:
    type: smac

critic:
  type: "SeqQMixer"
  model:
    model_type: QMixModel
    state_shape: 32
    input_dim: 5
    qmix_hidden_dim: 16
    hypernet_layers: 1
    hyper_hidden_dim: 16
  feature_extractor:
    model_type: "ResAttMaskedStateEnc"
    input_dim: 173
    embed_dim: 32
    num_heads: 2
    max_seq_len: 5
    dropout: 0.25
  seq_model:
    model_type: "SimpleResAttSeqEnc"
    input_dim: 32
    embed_dim: 32
    output_dim: 32
    num_heads: 1
    max_seq_len: 5
    dropout: 0.25
  optimizer:
    type: "Adam"
    lr: 0.0005
    weight_decay: 0.00005
  lr_scheduler:
    type: "ReduceLROnPlateau"
    mode: "max"
    patience: 3

rollout:
  manager_type: "persistent-env"
  worker_type: "persistent-env"
  n_workers: 1
  n_episodes: 1
  n_eval_episodes: 1
  traj_len: 5
  episode_limit: 2
  device: "cpu"
  victory_checker: smac

replay_buffer:
  type: "Prioritized"
  capacity: 2
  traj_len: 5
  priority_attr: "all_agents_sum_rewards"

analyzer:
  type: "default"

trainer:
  type: "MsgAggr"
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
"""

    def setUp(self):
        self.config = yaml.safe_load(self.yaml_config)

        if torch.cuda.is_available():
            self.config["trainer"]["train_device"] = "cuda"
            self.config["rollout"]["device"] = "cuda"

    def _create_trainer(self, temp_dir, config=None):
        if config is None:
            config = self.config
        config['trainer']['workdir'] = temp_dir
        return TrainerConfig(config).create_trainer()

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
            origin_agent_group_params = deepcopy(
                self.trainer.target_agent_group.state_dict()
            )
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=1)
            self.trainer._update_target_after_batch()
            critic_params = self.trainer.target_critic.state_dict()
            agent_group_params = self.trainer.target_agent_group.state_dict()

            for (w_name, w1), w2 in zip(
                critic_params.items(), origin_critic_params.values()
            ):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))

            for name in agent_group_params:
                if name in origin_agent_group_params:
                    if isinstance(agent_group_params[name], torch.Tensor):
                        if agent_group_params[name].requires_grad:
                            self.assertFalse(
                                torch.equal(
                                    agent_group_params[name],
                                    origin_agent_group_params[name],
                                )
                            )

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
        gpu_count = torch.cuda.device_count()
        if gpu_count < 2:
            self.skipTest(f"Need at least 2 GPUs for multi-GPU test, found {gpu_count}")
        config = deepcopy(self.config)
        config['trainer']['train_device'] = [f"cuda:{i}" for i in range(gpu_count)]
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self._create_trainer(temp_dir, config=config)
            origin_critic_params = deepcopy(self.trainer.eval_critic.state_dict())
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=1)
            self.trainer._update_target_after_batch()
            critic_params = self.trainer.eval_critic.state_dict()

            for w1, w2 in zip(critic_params.values(), origin_critic_params.values()):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))

    def test_torch_compile(self):
        config = deepcopy(self.config)
        config['trainer']['compile_models'] = True
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self._create_trainer(temp_dir, config=config)
            origin_critic_params = deepcopy(self.trainer.target_critic.state_dict())
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=1)
            self.trainer._update_target_after_batch()
            critic_params = self.trainer.target_critic.state_dict()

            for w1, w2 in zip(critic_params.values(), origin_critic_params.values()):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))


class TestProbSeqMsgAggrSMACQMIXTrainer(unittest.TestCase):
    yaml_config = """
agent_group:
  type: "ProbSeqMsgAggr"
  deterministic_eval: false
  agent_list:
    stalker_0: model_0
    stalker_1: model_0
    zealot_0: model_1
    zealot_1: model_1
    zealot_2: model_1
  models:
    model_0:
      feature_extractor:
        model_type: "Custom"
        layers:
        - type: Linear
          in_features: 144
          out_features: 16
      encoder:
        model_type: "SimpleResAttSeqEnc"
        input_dim: 16
        embed_dim: 16
        output_dim: 16
        num_heads: 1
        max_seq_len: 5
        dropout: 0.25
      decoder:
        model_type: "Custom"
        layers:
        - type: Linear
          in_features: 32
          out_features: 11
    model_1:
      feature_extractor:
        model_type: "Custom"
        layers:
        - type: Linear
          in_features: 144
          out_features: 16
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
        - type: Linear
          in_features: 16
          out_features: 16
        - type: ReLU
      decoder:
        model_type: "Custom"
        layers:
        - type: Linear
          in_features: 32
          out_features: 11
  aggr_model_config:
    model_type: "Custom"
    layers:
        - type: SelfAttention
          embed_dim: 16
          num_heads: 1
          batch_first: true
        - type: Permute
          dims: [0, 2, 1]
        - type: AdaptiveAvgPool1d
          output_size: 1
        - type: Flatten
        - type: Linear
          in_features: 16
          out_features: 32
        - type: ReLU
  optimizer:
    type: "Adam"
    lr: 0.0005
    weight_decay: 0.00005
  lr_scheduler:
    type: "ReduceLROnPlateau"
    mode: "max"
    patience: 3

environment:
  module_name: "smac_pettingzoo"
  env_name: "smacv1_pettingzoo_v1"
  env_params:
    map_name: "2s3z"
  wrapper:
    type: smac

critic:
  type: "ProbSeqQMixer"
  deterministic_eval: false
  model:
    model_type: QMixModel
    state_shape: 16
    input_dim: 5
    qmix_hidden_dim: 16
    hypernet_layers: 1
    hyper_hidden_dim: 16
  seq_model:
    model_type: "SimpleResAttSeqEnc"
    input_dim: 16
    embed_dim: 32
    output_dim: 32
    num_heads: 1
    max_seq_len: 5
    dropout: 0.0
  feature_extractor:
    model_type: "ResAttMaskedStateEnc"
    input_dim: 173
    embed_dim: 16
    num_heads: 1
    max_seq_len: 5
    dropout: 0.25
  optimizer:
    type: "Adam"
    lr: 0.0005
    weight_decay: 0.00005
  lr_scheduler:
    type: "ReduceLROnPlateau"
    mode: "max"
    patience: 3

rollout:
  manager_type: "persistent-env"
  worker_type: "persistent-env"
  n_workers: 1
  n_episodes: 1
  n_eval_episodes: 1
  traj_len: 5
  episode_limit: 2
  device: "cpu"
  victory_checker: smac

replay_buffer:
  type: "Prioritized"
  capacity: 2
  traj_len: 5
  priority_attr: "all_agents_sum_rewards"

analyzer:
  type: "default"

trainer:
  type: "ProbMsgAggr"
  gamma: 0.95
  eval_epsilon: 0.01
  workdir: "./test/results/replace_by_tempfile"
  train_device: "cpu"
  n_workers: 0
  sample_mode: "direct"

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
"""

    def setUp(self):
        self.config = yaml.safe_load(self.yaml_config)

        if torch.cuda.is_available():
            self.config["trainer"]["train_device"] = "cuda"
            self.config["rollout"]["device"] = "cuda"

    def _create_trainer(self, temp_dir, config=None):
        if config is None:
            config = self.config
        config['trainer']['workdir'] = temp_dir
        return TrainerConfig(config).create_trainer()

    def test_deterministic_eval(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self._create_trainer(temp_dir)
            self.assertFalse(self.trainer.eval_agent_group.deterministic_eval)
            self.assertFalse(self.trainer.eval_critic.deterministic_eval)

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
            origin_agent_group_params = deepcopy(
                self.trainer.target_agent_group.state_dict()
            )
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=1)
            self.trainer._update_target_after_batch()
            critic_params = self.trainer.target_critic.state_dict()
            agent_group_params = self.trainer.target_agent_group.state_dict()

            for (w_name, w1), w2 in zip(
                critic_params.items(), origin_critic_params.values()
            ):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))

            for name in agent_group_params:
                if name in origin_agent_group_params:
                    if isinstance(agent_group_params[name], torch.Tensor):
                        if agent_group_params[name].requires_grad:
                            self.assertFalse(
                                torch.equal(
                                    agent_group_params[name],
                                    origin_agent_group_params[name],
                                )
                            )

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
        gpu_count = torch.cuda.device_count()
        if gpu_count < 2:
            self.skipTest(f"Need at least 2 GPUs for multi-GPU test, found {gpu_count}")
        config = deepcopy(self.config)
        config['trainer']['train_device'] = [f"cuda:{i}" for i in range(gpu_count)]
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self._create_trainer(temp_dir, config=config)
            origin_critic_params = deepcopy(self.trainer.eval_critic.state_dict())
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=1)
            self.trainer._update_target_after_batch()
            critic_params = self.trainer.eval_critic.state_dict()

            for w1, w2 in zip(critic_params.values(), origin_critic_params.values()):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))

    def test_torch_compile(self):
        config = deepcopy(self.config)
        config['trainer']['compile_models'] = True
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self._create_trainer(temp_dir, config=config)
            origin_critic_params = deepcopy(self.trainer.target_critic.state_dict())
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=1)
            self.trainer._update_target_after_batch()
            critic_params = self.trainer.target_critic.state_dict()

            for w1, w2 in zip(critic_params.values(), origin_critic_params.values()):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))


class TestDualPathObsMsgAggrSMACQMIXTrainer(unittest.TestCase):
    yaml_config = """
agent_group:
  type: "DualPathObsMsgAggr"
  agent_list:
    stalker_0: model_0
    stalker_1: model_0
    zealot_0: model_1
    zealot_1: model_1
    zealot_2: model_1
  models:
    model_0:
      feature_extractor:
        model_type: "Custom"
        layers:
        - type: Linear
          in_features: 144
          out_features: 16
      msg_feature_extractor:
        model_type: "Custom"
        layers:
        - type: Linear
          in_features: 144
          out_features: 16
      encoder:
        model_type: "SimpleResAttSeqEnc"
        input_dim: 16
        embed_dim: 16
        output_dim: 16
        num_heads: 1
        max_seq_len: 5
        dropout: 0.25
      decoder:
        model_type: "Custom"
        layers:
        - type: Linear
          in_features: 32
          out_features: 11
    model_1:
      feature_extractor:
        model_type: "Custom"
        layers:
        - type: Linear
          in_features: 144
          out_features: 16
      msg_feature_extractor:
        model_type: "Custom"
        layers:
        - type: Linear
          in_features: 144
          out_features: 16
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
        - type: Linear
          in_features: 16
          out_features: 16
        - type: ReLU
      decoder:
        model_type: "Custom"
        layers:
        - type: Linear
          in_features: 32
          out_features: 11
  aggr_model_config:
    model_type: "Custom"
    layers:
        - type: SelfAttention
          embed_dim: 16
          num_heads: 1
          batch_first: true
        - type: Permute
          dims: [0, 2, 1]
        - type: AdaptiveAvgPool1d
          output_size: 1
        - type: Flatten
        - type: Linear
          in_features: 16
          out_features: 16
  optimizer:
    type: "Adam"
    lr: 0.0005
    weight_decay: 0.00005
  lr_scheduler:
    type: "ReduceLROnPlateau"
    mode: "max"
    patience: 3

environment:
  module_name: "smac_pettingzoo"
  env_name: "smacv1_pettingzoo_v1"
  env_params:
    map_name: "2s3z"
  wrapper:
    type: smac

critic:
  type: "QMixer"
  model:
    model_type: QMixModel
    state_shape: 16
    input_dim: 5
    qmix_hidden_dim: 16
    hypernet_layers: 1
    hyper_hidden_dim: 16
  feature_extractor:
    model_type: "ResAttMaskedStateEnc"
    input_dim: 173
    embed_dim: 16
    num_heads: 4
    max_seq_len: 5
    dropout: 0.25
  optimizer:
    type: "Adam"
    lr: 0.0005
    weight_decay: 0.00005
  lr_scheduler:
    type: "ReduceLROnPlateau"
    mode: "max"
    patience: 3

rollout:
  manager_type: "multi-process"
  worker_type: "multi-process"
  n_workers: 1
  n_episodes: 1
  n_eval_episodes: 1
  traj_len: 5
  episode_limit: 16
  device: "cpu"
  victory_checker: smac

replay_buffer:
  type: "Prioritized"
  capacity: 4
  traj_len: 5
  priority_attr: "all_agents_sum_rewards"

analyzer:
  type: "default"

trainer:
  type: "MsgAggr"
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
"""

    def setUp(self):
        self.config = yaml.safe_load(self.yaml_config)

        if torch.cuda.is_available():
            self.config["trainer"]["train_device"] = "cuda"
            self.config["rollout"]["device"] = "cuda"

    def _create_trainer(self, temp_dir, config=None):
        if config is None:
            config = self.config
        config['trainer']['workdir'] = temp_dir
        return TrainerConfig(config).create_trainer()

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
            origin_agent_group_params = deepcopy(
                self.trainer.target_agent_group.state_dict()
            )
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=1)
            self.trainer._update_target_after_batch()
            critic_params = self.trainer.target_critic.state_dict()
            agent_group_params = self.trainer.target_agent_group.state_dict()

            for (w_name, w1), w2 in zip(
                critic_params.items(), origin_critic_params.values()
            ):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))

            for name in agent_group_params:
                if name in origin_agent_group_params:
                    if isinstance(agent_group_params[name], torch.Tensor):
                        if agent_group_params[name].requires_grad:
                            self.assertFalse(
                                torch.equal(
                                    agent_group_params[name],
                                    origin_agent_group_params[name],
                                )
                            )

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
        gpu_count = torch.cuda.device_count()
        if gpu_count < 2:
            self.skipTest(f"Need at least 2 GPUs for multi-GPU test, found {gpu_count}")
        config = deepcopy(self.config)
        config['trainer']['train_device'] = [f"cuda:{i}" for i in range(gpu_count)]
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self._create_trainer(temp_dir, config=config)
            origin_critic_params = deepcopy(self.trainer.eval_critic.state_dict())
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=1)
            self.trainer._update_target_after_batch()
            critic_params = self.trainer.eval_critic.state_dict()

            for w1, w2 in zip(critic_params.values(), origin_critic_params.values()):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))

    def test_torch_compile(self):
        config = deepcopy(self.config)
        config['trainer']['compile_models'] = True
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self._create_trainer(temp_dir, config=config)
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
