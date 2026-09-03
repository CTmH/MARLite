import unittest
import os
import yaml
from copy import deepcopy
import tempfile
import torch

from marlite.trainer import TrainerConfig, REGISTERED_TRAINERS


class TestAEGroupConsensusTrainer(unittest.TestCase):
    yaml_config = """\
agent_group:
  type: "GroupConsensusQMIX"
  deterministic_eval: true
  enable_rl_grad_to_group_estimate: false
  consensus_mode: "ae"
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
      group_estimate_feature_extractor:
        model_type: "Custom"
        layers:
        - type: ChannelSelector
          num_channels: [0, 1, 2, 4, 5, 39, 40]
          dim: -3
        - type: Conv2d
          in_channels: 7
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
          out_features: 256
        - type: LayerNorm
          normalized_shape: 256
        - type: GELU
        - type: Linear
          in_features: 256
          out_features: 64
      feature_extractor:
        model_type: "Custom"
        layers:
        - type: Conv2d
          in_channels: 41
          out_channels: 16
          kernel_size: 1
          stride: 1
          padding: 0
        - type: BatchNorm2d
          num_features: 16
        - type: GELU
        - type: Conv2d
          in_channels: 16
          out_channels: 8
          kernel_size: 1
          stride: 1
          padding: 0
        - type: BatchNorm2d
          num_features: 8
        - type: GELU
        - type: Flatten
        - type: Linear
          in_features: 1352
          out_features: 256
        - type: LayerNorm
          normalized_shape: 256
        - type: GELU
        - type: Linear
          in_features: 256
          out_features: 64
      encoder:
        model_type: "SimpleResAttSeqEnc"
        input_dim: 64
        embed_dim: 32
        output_dim: 64
        num_heads: 2
        max_seq_len: 8
        dropout: 0.0
      decoder:
        model_type: "Custom"
        layers:
        - type: LayerNorm
          normalized_shape: 128
        - type: GELU
        - type: Linear
          in_features: 128
          out_features: 21
  group_builder:
    type: "Fixed"
    group_ids: [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2]
  optimizer:
    type: "Adam"
    lr: 0.001
    weight_decay: 0.000001
  lr_scheduler:
    type: "ReduceLROnPlateau"
    mode: "max"
    factor: 0.5
    patience: 3
    min_lr: 0.000001

environment:
  module_name: "magent2.environments"
  env_name: "battle_v4"
  env_params:
    map_size: 32
    step_reward: -0.0001
    dead_penalty: -1.0
    attack_penalty: -0.001
    attack_opponent_reward: 0.8
    extra_features: true
    minimap_mode: true
  wrapper:
    type: battle
    opp_obs_queue_len: 1
    channel_first: true
    vector_state: true
    vector_observation: false
    opponent_agent_group:
      type: "MAgentBattle"
      strategy: "advanced"
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
  type: "SeqQMixer"
  model:
    model_type: QMixModel
    state_shape: 32
    input_dim: 36
    qmix_hidden_dim: 32
    hypernet_layers: 2
    hyper_hidden_dim: 32
  feature_extractor:
    model_type: "SimpleResAttStateEnc"
    input_dim: 62
    embed_dim: 32
    num_heads: 4
    max_seq_len: 72
    dropout: 0.0
  seq_model:
    model_type: "SimpleResAttSeqEnc"
    input_dim: 32
    embed_dim: 32
    output_dim: 32
    num_heads: 2
    max_seq_len: 8
    dropout: 0.0
  optimizer:
    type: "Adam"
    lr: 0.001
    weight_decay: 0.000001
  lr_scheduler:
    type: "ReduceLROnPlateau"
    mode: "max"
    factor: 0.5
    patience: 8
    min_lr: 0.000001

self_supervised_learning:
  model:
    model_type: "Custom"
    layers:
    - type: ResidualMLP
      input_dim: 64
      hidden_dim: 64
    - type: LayerNorm
      normalized_shape: 64
    - type: GELU
    - type: Dropout
      p: 0.25
    - type: Linear
      in_features: 64
      out_features: 17
  optimizer:
    type: "Adam"
    lr: 0.001
    weight_decay: 0.000001
  lr_scheduler:
    type: "ReduceLROnPlateau"
    mode: "max"
    factor: 0.5
    patience: 8
    min_lr: 0.000001
  data_constructor:
    type: "MagentVecStateGroupFeatures"
    observation_range: 0.40625
    n_groups: 3
    coord_dims: [3, 4]
    hp_dim: 0
    team_dim: 1
    my_team: 0
    enemy_team: 1
    action_dim: 21
    n_offsets: 6
    n_workers: 0
  reconstruction_loss:
    type: "PointSetMSE"

rollout:
  manager_type: "persistent-env"
  worker_type: "persistent-env"
  n_workers: 1
  n_episodes: 1
  n_eval_episodes: 1
  traj_len: 8
  episode_limit: 2
  victory_checker: "battle_wrapper"
  device: "cpu"

replay_buffer:
  type: "Prioritized"
  capacity: 2
  traj_len: 8
  priority_attr: all_agents_sum_rewards
  alpha: 0.45

analyzer:
  type: "default"

trainer:
  type: "SSLGroupConsensusQMIX"
  gamma: 0.95
  eval_epsilon: 0.001
  update_cache_threshold: 0.01
  eval_episodes_to_replay_ratio: 0.125
  workdir: "./test/results/replace_by_tempfile"
  train_device: "cuda:0"
  compile_models: false
  sample_mode: "direct"
  n_workers: 0
  kl_divergence_weight: 0
  target_update_interval: 1
  kl_on_agent: false
  kl_on_group: false
  consensus_mode: "ae"
  loss_combination_method: "pit_loss"
  warmup_epochs: 0

  epsilon_scheduler:
    type: "linear"
    start_value: 1.0
    end_value: 0.05
    ramp_start_step: 0
    ramp_steps: 10

  sample_ratio_scheduler:
    type: "linear"
    start_value: 16
    end_value: 16
    ramp_start_step: 0
    ramp_steps: 10

  train_args:
    epochs: 1
    target_first_metric: 10000000
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

    def test_distributed_data_parallel(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping multi-GPU test")

        gpu_count = torch.cuda.device_count()
        if gpu_count < 2:
            self.skipTest(f"Need at least 2 GPUs for multi-GPU test, found {gpu_count}")

        config = deepcopy(self.config)
        device_list = [f"cuda:{i}" for i in range(gpu_count)]
        config['trainer']['train_device'] = device_list

        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self._create_trainer(temp_dir, config=config)

            self.assertTrue(self.trainer.use_multi_gpu)
            self.assertEqual(len(self.trainer.device_list), gpu_count)
            self.assertIsNotNone(self.trainer.worker_group)
            self.assertEqual(self.trainer.worker_group.world_size, gpu_count)

            origin_critic_params = deepcopy(self.trainer.target_critic.state_dict())
            origin_agent_group_params = deepcopy(
                self.trainer.target_agent_group.state_dict()
            )
            origin_ssl_model_params = deepcopy(self.trainer.ssl_model.state_dict())

            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=1)
            self.trainer._update_target_after_batch()

            critic_params = self.trainer.target_critic.state_dict()
            agent_group_params = self.trainer.target_agent_group.state_dict()
            ssl_model_params = self.trainer.ssl_model.state_dict()

            for w1, w2 in zip(critic_params.values(), origin_critic_params.values()):
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

            for name in ssl_model_params:
                if name in origin_ssl_model_params:
                    if isinstance(ssl_model_params[name], torch.Tensor):
                        if ssl_model_params[name].requires_grad:
                            self.assertFalse(
                                torch.equal(
                                    ssl_model_params[name],
                                    origin_ssl_model_params[name],
                                )
                            )


if __name__ == "__main__":
    unittest.main()
