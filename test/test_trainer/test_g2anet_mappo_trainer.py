import unittest
import os
import yaml
from copy import deepcopy
import tempfile
import torch

from marlite.trainer import TrainerConfig


class TestGraphMAPPOTrainer(unittest.TestCase):
    yaml_config = """
agent_group:
  type: "G2ANetMAPPO"
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
    opp_obs_queue_len: 1
    channel_first: true

critic:
  type: "MAPPOCritic"
  model:
    model_type: "Custom"
    layers:
      - type: Linear
        in_features: 256
        out_features: 128
      - type: LeakyReLU
      - type: Linear
        in_features: 128
        out_features: 1
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
  type: "Normal"
  capacity: 2
  traj_len: 1

analyzer:
  type: "default"

trainer:
  type: "GraphMAPPO"
  gamma: 0.95
  clip_epsilon: 0.2
  gae_lambda: 0.95
  entropy_coef: 0.01
  vf_coef: 0.5
  max_grad_norm: 5.0
  workdir: "./test/results/replace_by_tempfile"
  train_device: "cpu"
  n_workers: 0

  sample_ratio_scheduler:
    type: "linear"
    start_value: 16
    end_value: 16
    ramp_start_step: 0
    ramp_steps: 10

  train_args:
    iterations: 1
    target_first_metric: 100
    batch_size: 8
    learning_times_per_iteration: 1
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
        trainer = TrainerConfig(config).create_trainer()
        trainer.logdir = os.path.join(temp_dir, "logs")
        trainer.checkpointdir = os.path.join(temp_dir, "checkpoints")
        return trainer

    def test_collect_experience(self):
        """On-policy: calling evaluate() fills the replay buffer."""
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = self._create_trainer(temp_dir)
            trainer.evaluate()
            self.assertNotEqual(len(trainer.replaybuffer.buffer), 0)

    def test_learn(self):
        """Verify parameters change after learning."""
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = self._create_trainer(temp_dir)
            trainer.evaluate()
            origin_critic = deepcopy(trainer.eval_critic.state_dict())
            origin_agent = deepcopy(trainer.eval_agent_group.state_dict())
            trainer.learn(sample_size=32, batch_size=8, times=1)
            critic_params = trainer.eval_critic.state_dict()
            agent_params = trainer.eval_agent_group.state_dict()
            for w1, w2 in zip(critic_params.values(), origin_critic.values()):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))
            for w1, w2 in zip(agent_params.values(), origin_agent.values()):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))

    def test_save_load_checkpoint(self):
        checkpoint = "test_checkpoint"
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = self._create_trainer(temp_dir)
            trainer.save_current_model(checkpoint)
            trainer.load_checkpoint(checkpoint)

    def test_train(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = self._create_trainer(temp_dir)
            result = trainer.evaluate()
            best_metrics = trainer.train(
                iterations=2, target_first_metric=100
            )

    def test_torch_compile(self):
        config = deepcopy(self.config)
        config["trainer"]["compile_models"] = True
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = self._create_trainer(temp_dir, config)
            trainer.evaluate()
            origin_critic = deepcopy(trainer.eval_critic.state_dict())
            trainer.learn(sample_size=32, batch_size=8, times=1)
            critic_params = trainer.eval_critic.state_dict()
            for w1, w2 in zip(critic_params.values(), origin_critic.values()):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))


if __name__ == "__main__":
    unittest.main()
