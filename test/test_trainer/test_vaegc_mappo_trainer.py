import unittest
import yaml
from copy import deepcopy
import tempfile
import torch
import pygame
import numpy as np
from unittest.mock import Mock

from marlite.trainer import TrainerConfig


class TestVAEGCMAPPOTrainer(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load("""
agent_group:
  type: "GroupConsensusMAPPO"
  deterministic_eval: true
  enable_rl_grad_to_group_estimate: false
  merge_mode: "bayesian"
  consensus_mode: "vae"
  agent_list:
    agent_0: model1
    agent_1: model1
    agent_2: model1
  models:
    model1:
      group_estimate_feature_extractor:
        model_type: "Identity"
      feature_extractor:
        model_type: "Identity"
      encoder:
        model_type: "Custom"
        layers:
          - type: "Linear"
            in_features: 18
            out_features: 55
          - type: "ReLU"
      decoder:
        model_type: "Custom"
        layers:
          - type: "Linear"
            in_features: 64
            out_features: 5
  group_builder:
    type: "Fixed"
    group_ids: [0, 0, 0]
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
        out_features: 32
      - type: "ReLU"
      - type: "Linear"
        in_features: 32
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

self_supervised_learning:
  model:
    model_type: "Custom"
    layers:
      - type: "Linear"
        in_features: 9
        out_features: 32
      - type: "ReLU"
      - type: "Linear"
        in_features: 32
        out_features: 18
  optimizer:
    type: "Adam"
    lr: 0.0005
    weight_decay: 0.0001
  lr_scheduler:
    type: "ReduceLROnPlateau"
    mode: "max"
    factor: 0.5
    patience: 8
    min_lr: 0.000001
  data_constructor:
    type: "MagentVecObs"
    max_entities_perception: 1
    n_workers: 0
  reconstruction_loss:
    type: "PointSetMSE"

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
  consensus_mode: "vae"
  kl_divergence_weight: 0.005
  recon_mode: "per_group"
  kl_on_agent: true
  kl_on_group: false
  warmup_iterations: 99999
  rl_consensus_gate_scheduler:
    type: "logarithmic"
    start_value: 0.0
    end_value: 1.0
    ramp_start_step: 2
    ramp_steps: 4
    curve_rate: 0.0

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
            self.trainer.collect_experience(0.9)
            self.assertNotEqual(len(self.trainer.replaybuffer.buffer), 0)

    def test_rl_consensus_gate_scheduler_is_injected_into_rollout_agent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = self._create_trainer(temp_dir)
            self.assertEqual(trainer.eval_agent_group.rl_consensus_gate.item(), 0.0)

            trainer._prepare_rollout(4)
            self.assertEqual(trainer.eval_agent_group.rl_consensus_gate.item(), 0.5)

            trainer._prepare_rollout(6)
            self.assertEqual(trainer.eval_agent_group.rl_consensus_gate.item(), 1.0)

    def test_ssl_update_mode_defaults_to_joint_and_accepts_sequential(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = self._create_trainer(temp_dir)
            self.assertEqual(trainer.ssl_update_mode, "joint")

        self.config["trainer"]["ssl_update_mode"] = "sequential"
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = self._create_trainer(temp_dir)
            self.assertEqual(trainer.ssl_update_mode, "sequential")

    def test_rollout_gate_schedule_updates_only_before_the_next_rollout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = self._create_trainer(temp_dir)
            gate_values = []
            prepare_rollout = trainer._prepare_rollout

            def record_gate(rollout_iteration):
                prepare_rollout(rollout_iteration)
                gate_values.append(
                    trainer.eval_agent_group.rl_consensus_gate.item()
                )

            result = {
                trainer.eval_metric_list[0]: {"mean": 0.0, "std": 0.0}
            }
            trainer._prepare_rollout = record_gate
            trainer.evaluate = Mock(return_value=result)
            trainer.learn = Mock(return_value={"loss": 0.0})
            trainer.save_intermediate_results = Mock()
            trainer.save_best_model = Mock()

            trainer.train(iterations=4, target_first_metric=1.0)

            self.assertEqual(gate_values, [0.0, 0.0, 0.0, 0.25, 0.5])
            self.assertEqual(trainer.evaluate.call_count, 5)

    def test_learn(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self._create_trainer(temp_dir)
            origin_actor_params = deepcopy(
                self.trainer.eval_agent_group.state_dict()
            )
            origin_critic_params = deepcopy(
                self.trainer.eval_critic.state_dict()
            )
            self.trainer.collect_experience(0.9)
            result = self.trainer.learn(sample_size=32, batch_size=8, times=1)
            self.assertIsInstance(result, dict)
            self.assertIsInstance(result["loss"], float)

            actor_params = self.trainer.eval_agent_group.state_dict()
            for key in actor_params:
                if actor_params[key].requires_grad:
                    self.assertFalse(
                        torch.equal(
                            actor_params[key], origin_actor_params[key]
                        ),
                        f"Actor param {key} did not change after learning",
                    )

            critic_params = self.trainer.eval_critic.state_dict()
            for key in critic_params:
                if critic_params[key].requires_grad:
                    self.assertFalse(
                        torch.equal(
                            critic_params[key], origin_critic_params[key]
                        ),
                        f"Critic param {key} did not change after learning",
                    )

    def test_sequential_learning_orders_and_isolates_updates(self):
        self.config["trainer"]["ssl_update_mode"] = "sequential"
        self.config["trainer"]["warmup_iterations"] = 0
        self.config["agent_group"]["models"]["model1"][
            "group_estimate_feature_extractor"
        ] = {
            "model_type": "Custom",
            "layers": [
                {
                    "type": "Linear",
                    "in_features": 18,
                    "out_features": 18,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = self._create_trainer(temp_dir)
            trainer.collect_experience(0.9)
            trainer.data_constructor = Mock()
            trainer.data_constructor.process.side_effect = (
                lambda observations, states, grouping, alive_mask: (
                    observations[:, -1].mean(axis=1)[:, None, :],
                    np.ones((observations.shape[0], 1), dtype=bool),
                )
            )
            trainer.n_workers = 0

            estimator_params = list(
                trainer.eval_agent_group.group_estimate_feature_extractors.parameters()
            )
            estimator_ids = {id(parameter) for parameter in estimator_params}
            policy_params = [
                parameter
                for parameter in trainer.eval_agent_group.parameters()
                if id(parameter) not in estimator_ids
            ]
            critic_params = list(trainer.eval_critic.parameters())
            initial_estimator = [
                parameter.detach().clone() for parameter in estimator_params
            ]
            initial_policy = [
                parameter.detach().clone() for parameter in policy_params
            ]
            phases = []
            original_ppo_step = trainer._ppo_step_single_gpu
            original_ssl_step = trainer._ssl_step_single_gpu

            def checked_ppo_step(batch):
                before = [parameter.detach().clone() for parameter in estimator_params]
                result = original_ppo_step(batch)
                self.assertTrue(
                    all(
                        torch.equal(parameter, old)
                        for parameter, old in zip(estimator_params, before)
                    )
                )
                phases.append("ppo")
                return result

            def checked_ssl_step(batch):
                before_policy = [
                    parameter.detach().clone() for parameter in policy_params
                ]
                before_critic = [
                    parameter.detach().clone() for parameter in critic_params
                ]
                result = original_ssl_step(batch)
                self.assertTrue(
                    all(
                        torch.equal(parameter, old)
                        for parameter, old in zip(policy_params, before_policy)
                    )
                )
                self.assertTrue(
                    all(
                        torch.equal(parameter, old)
                        for parameter, old in zip(critic_params, before_critic)
                    )
                )
                phases.append("ssl")
                return result

            trainer._ppo_step_single_gpu = checked_ppo_step
            trainer._ssl_step_single_gpu = checked_ssl_step
            result = trainer.learn(
                sample_size=16, batch_size=8, times=2, ssl_times=1
            )
            self.assertIsInstance(result, dict)
            self.assertIsInstance(result["loss"], float)

            first_ssl = phases.index("ssl")
            self.assertTrue(all(phase == "ppo" for phase in phases[:first_ssl]))
            self.assertTrue(all(phase == "ssl" for phase in phases[first_ssl:]))
            self.assertTrue(
                any(
                    not torch.equal(parameter, old)
                    for parameter, old in zip(estimator_params, initial_estimator)
                )
            )
            self.assertTrue(
                any(
                    not torch.equal(parameter, old)
                    for parameter, old in zip(policy_params, initial_policy)
                )
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


if __name__ == "__main__":
    unittest.main()
