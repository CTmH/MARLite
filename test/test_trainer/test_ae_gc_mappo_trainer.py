import unittest
import os
import yaml
from copy import deepcopy
import tempfile
import torch
import pygame

from marlite.trainer import TrainerConfig


class TestAEGCMAPPOTrainer(unittest.TestCase):
    yaml_config = """
agent_group:
  type: "GroupConsensusMAPPO"
  deterministic_eval: true
  enable_rl_grad_to_group_estimate: false
  consensus_mode: "ae"
  merge_mode: "bayesian"
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
  type: "MAPPOCritic"
  model:
    model_type: "Custom"
    layers:
      - type: "Linear"
        in_features: 32
        out_features: 64
      - type: "ReLU"
      - type: "Linear"
        in_features: 64
        out_features: 1
  feature_extractor:
    model_type: "SimpleResAttStateEnc"
    input_dim: 62
    embed_dim: 32
    num_heads: 2
    max_seq_len: 72
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
  episode_limit: 5
  victory_checker: "battle_wrapper"
  device: "cpu"

replay_buffer:
  type: "Prioritized"
  capacity: 5
  traj_len: 8
  priority_attr: all_agents_sum_rewards
  alpha: 0.45

analyzer:
  type: "default"

trainer:
  type: "SSLGroupConsensusMAPPO"
  gamma: 0.99
  workdir: "./test/results/replace_by_tempfile"
  train_device: "cpu"
  clip_epsilon: 0.2
  gae_lambda: 0.95
  entropy_coef: 0.01
  vf_coef: 0.5
  max_grad_norm: 5.0
  consensus_mode: "ae"
  kl_divergence_weight: 0
  recon_mode: "per_group"
  kl_on_agent: false
  kl_on_group: false
  warmup_iterations: 0
  n_workers: 0
  sample_mode: "direct"

  sample_ratio_scheduler:
    type: "linear"
    start_value: 16
    end_value: 16
    ramp_start_step: 0
    ramp_steps: 10

  train_args:
    iterations: 1
    target_first_metric: -9999
    batch_size: 8
    learning_times_per_iteration: 4
"""

    def setUp(self):
        self.config = yaml.safe_load(self.yaml_config)

        if torch.cuda.is_available():
            self.config["trainer"]["train_device"] = "cuda"
            self.config["rollout"]["device"] = "cuda"

    def _create_trainer(self, temp_dir, config=None):
        if config is None:
            config = self.config
        config["trainer"]["workdir"] = temp_dir
        return TrainerConfig(config).create_trainer()

    def test_collect_experience(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = self._create_trainer(temp_dir)
            trainer.collect_experience(0.9)
            self.assertNotEqual(len(trainer.replaybuffer.buffer), 0)

    def test_learn(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = self._create_trainer(temp_dir)
            origin_actor_params = deepcopy(
                trainer.eval_agent_group.state_dict()
            )
            origin_critic_params = deepcopy(
                trainer.eval_critic.state_dict()
            )
            trainer.collect_experience(0.9)
            trainer.learn(sample_size=32, batch_size=8, times=1)

            actor_params = trainer.eval_agent_group.state_dict()
            for key in actor_params:
                if actor_params[key].requires_grad:
                    self.assertFalse(
                        torch.equal(
                            actor_params[key], origin_actor_params[key]
                        ),
                        f"Actor param {key} did not change after learning",
                    )

            critic_params = trainer.eval_critic.state_dict()
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
            trainer = self._create_trainer(temp_dir)
            trainer.save_current_model(checkpoint)
            trainer.load_checkpoint(checkpoint)

    def test_train(self):
        pygame.font.init()
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = self._create_trainer(temp_dir)
            result = trainer.evaluate()
            best_metrics = trainer.train(
                iterations=2, target_first_metric=5
            )


if __name__ == "__main__":
    unittest.main()
