import unittest
import yaml
from copy import deepcopy
import tempfile
import torch

from marlite.trainer import TrainerConfig


class TestVAEGraphQMIXBattle(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load("""
agent_group:
  type: "ProbObsGNNComm"
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
        model_type: "SimpleResAttObsEnc"
        input_dim: 43
        embed_dim: 16
        output_dim: 16
        num_heads: 1
        max_seq_len: 8
      encoder:
        model_type: "SimpleResAttSeqEnc"
        input_dim: 16
        embed_dim: 16
        output_dim: 16
        num_heads: 2
        max_seq_len: 8
        dropout: 0.0
      decoder:
        model_type: "Custom"
        layers:
        - type: ReLU
        - type: Linear
          in_features: 32
          out_features: 21
  graph_builder:
    type: "FullConn"
    valid_node_list: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
    add_self_loop: true
  graph_model:
    model_type: "GAT"
    input_dim: 16
    hidden_dim: 16
    output_dim: 32
    head_conv1: 1
    head_conv2: 1
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
    minimap_mode: true
  wrapper:
    type: battle
    opp_obs_queue_len: 1
    channel_first: true
    vector_state: true
    vector_observation: true
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
    state_shape: 16
    input_dim: 36
    qmix_hidden_dim: 16
    hypernet_layers: 1
    hyper_hidden_dim: 16
  feature_extractor:
    model_type: "SimpleResAttStateEnc"
    input_dim: 62
    embed_dim: 16
    num_heads: 1
    max_seq_len: 72
    dropout: 0.0
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

self_supervised_learning:
  model:
    model_type: "Custom"
    layers:
    - type: Linear
      in_features: 16
      out_features: 176
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
  data_constructor:
    type: "MagentVecObs"
    max_entities_perception: 16
    n_workers: 8
    with_time_seq: false
    include_features: [ 0, 1, 2, 3, 4, 5, 6, 39, 40, 41, 42]
  reconstruction_loss:
    type: "ChamferDist"
    use_squared_distance: True

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
  type: "VAEGraphQMIX"
  gamma: 0.95
  eval_epsilon: 0.005
  eval_episodes_to_replay_ratio: 0.25
  workdir: "./test/results/replace_by_tempfile"
  train_device: "cpu"
  compile_models: false
  n_workers: 0
  self_supervised_learning_loss_weight: 0.75
  target_update_interval: 2

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
    batch_size: 8
    learning_times_per_epoch: 1
    ssl_batch_size: 256
    ssl_learning_times_per_epoch: 1
""")

        if torch.cuda.is_available():
            self.config["trainer"]["train_device"] = "cuda"
            self.config["rollout"]["device"] = "cuda"

    def _create_trainer(self, temp_dir):
        self.config['trainer']['workdir'] = temp_dir
        return TrainerConfig(self.config).create_trainer()

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
        checkpoint = 'test_checkpoint'
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self._create_trainer(temp_dir)
            self.trainer.save_current_model(checkpoint)
            self.trainer.load_checkpoint(checkpoint)

    def test_train(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self._create_trainer(temp_dir)
            result = self.trainer.evaluate()
            best_metrics = self.trainer.train(epochs=2, target_first_metric=5)
