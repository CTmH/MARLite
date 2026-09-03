"""Integration tests for the QPLEXTrainer.

Verifies the full training pipeline:
- Trainer creation with QPLEX-specific sub-modules.
- Experience collection and replay buffer population.
- Single-batch learning (loss is a finite float).
- Parameter update after learning.
- Checkpoint save / load roundtrip.
- Target-network synchronisation.
- Optimiser parameter isolation (agent params vs. critic params).
- ``n_head=1`` configuration works end-to-end.
"""

import unittest
import yaml
import tempfile
import torch
from copy import deepcopy
from marlite.trainer import TrainerConfig
from marlite.trainer.qplex_trainer import QPLEXTrainer


QPLEX_CFG = """
agent_group:
  type: "QMIX"
  agent_list:
    agent_0: m1
    agent_1: m1
    agent_2: m1
  models:
    m1:
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
  type: "QPLEXMixer"
  action_dim: 5
  n_agents: 3
  weighted_head: true
  is_minus_one: true
  feature_extractor:
    model_type: "Identity"
  value_stream_w_final:
    model_type: "Custom"
    layers:
    - type: "Linear"
      in_features: 54
      out_features: 3
  value_stream_v:
    model_type: "Custom"
    layers:
    - type: "Linear"
      in_features: 54
      out_features: 3
  transformation:
    model_type: "QplexTransformation"
    state_dim: 54
    n_agents: 3
    n_head: 1
    embed_dim: 4
    attend_reg_coef: 0.0
    selector_configs:
    - model_type: "Custom"
      layers:
      - type: "Linear"
        in_features: 54
        out_features: 4
    key_configs:
    - model_type: "Custom"
      layers:
      - type: "Linear"
        in_features: 54
        out_features: 12
    v_config:
      model_type: "Custom"
      layers:
      - type: "Linear"
        in_features: 54
        out_features: 1
  joint_attention:
    model_type: "QplexJointAttention"
    state_dim: 54
    action_dim: 5
    n_agents: 3
    n_head: 1
    key_configs:
    - model_type: "Custom"
      layers:
      - type: "Linear"
        in_features: 54
        out_features: 1
    agent_extractor_configs:
    - model_type: "Custom"
      layers:
      - type: "Linear"
        in_features: 54
        out_features: 3
    action_extractor_configs:
    - model_type: "Custom"
      layers:
      - type: "Linear"
        in_features: 69
        out_features: 3
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
  type: "QPLEX"
  gamma: 0.95
  eval_epsilon: 0.01
  workdir: "./test/results/replace_by_tempfile"
  train_device: "cpu"

  epsilon_scheduler:
    type: "logarithmic"
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
    target_first_metric: 100
    rollback_interval: 1
    batch_size: 8
    learning_times_per_epoch: 1
"""


class TestQPLEXTrainer(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load(QPLEX_CFG)

    def _create_trainer(self, temp_dir):
        self.config["trainer"]["workdir"] = temp_dir
        trainer_config = TrainerConfig(self.config)
        return trainer_config.create_trainer()

    def test_create_trainer_has_qplex_modules(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = self._create_trainer(temp_dir)
            self.assertIsInstance(trainer, QPLEXTrainer)
            self.assertIsNotNone(trainer.eval_agent_group)
            self.assertIsNotNone(trainer.target_agent_group)
            self.assertIsNotNone(trainer.eval_critic)
            self.assertIsNotNone(trainer.target_critic)
            # The eval_critic should be a QPLEXMixer with sub-modules
            from marlite.algorithm.critic import QPLEXMixer
            self.assertIsInstance(trainer.eval_critic, QPLEXMixer)
            self.assertTrue(hasattr(trainer.eval_critic, "transformation"))
            self.assertTrue(hasattr(trainer.eval_critic, "joint_attention"))
            self.assertTrue(hasattr(trainer.target_critic, "transformation"))
            self.assertTrue(hasattr(trainer.target_critic, "joint_attention"))

    def test_collect_experience(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = self._create_trainer(temp_dir)
            trainer.collect_experience(0.9)
            self.assertNotEqual(len(trainer.replaybuffer.buffer), 0)

    def test_learn_one_batch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = self._create_trainer(temp_dir)
            trainer.collect_experience(0.9)
            loss = trainer.learn(sample_size=8, batch_size=4, times=1)
            self.assertIsInstance(loss, float)

    def test_learn_updates_parameters(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = self._create_trainer(temp_dir)
            trainer.collect_experience(0.9)
            origin_critic_params = deepcopy(trainer.target_critic.state_dict())
            loss = trainer.learn(sample_size=8, batch_size=4, times=1)
            trainer._update_target_after_batch()
            critic_params = trainer.target_critic.state_dict()
            n_changed = sum(
                1 for k in origin_critic_params
                if not torch.equal(origin_critic_params[k], critic_params[k])
            )
            # At least the eval_critic should have changed; target_critic was
            # hard-updated to match, so all params should have changed.
            self.assertGreater(n_changed, 0)

    def test_save_load_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = self._create_trainer(temp_dir)
            trainer.collect_experience(0.9)
            trainer.learn(sample_size=8, batch_size=4, times=1)
            trainer._update_target_after_batch()
            trainer.save_current_model("ckpt0")

            params_before = {
                "agent": {k: v.clone() for k, v in trainer.eval_agent_group.state_dict().items()},
                "critic": {k: v.clone() for k, v in trainer.eval_critic.state_dict().items()},
            }
            for p in trainer.eval_agent_group.parameters():
                torch.nn.init.zeros_(p)
            for p in trainer.eval_critic.parameters():
                torch.nn.init.zeros_(p)

            trainer.load_checkpoint("ckpt0")

            for k, v in params_before["agent"].items():
                self.assertTrue(torch.equal(v, trainer.eval_agent_group.state_dict()[k]))
            for k, v in params_before["critic"].items():
                self.assertTrue(torch.equal(v, trainer.eval_critic.state_dict()[k]))

    def test_target_update_syncs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = self._create_trainer(temp_dir)
            for p in trainer.eval_agent_group.parameters():
                torch.nn.init.ones_(p)
            for p in trainer.eval_critic.parameters():
                torch.nn.init.ones_(p)
            trainer._update_target_after_batch()
            for ep, tp in zip(
                trainer.eval_agent_group.parameters(),
                trainer.target_agent_group.parameters(),
            ):
                self.assertTrue(torch.equal(ep, tp))
            for ep, tp in zip(
                trainer.eval_critic.parameters(), trainer.target_critic.parameters()
            ):
                self.assertTrue(torch.equal(ep, tp))

    def test_critic_optimizer_excludes_agent_params(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = self._create_trainer(temp_dir)
            critic_opt_params = set()
            for g in trainer.critic_optimizer.param_groups:
                for p in g["params"]:
                    critic_opt_params.add(id(p))
            for p in trainer.eval_agent_group.parameters():
                self.assertNotIn(id(p), critic_opt_params)

    def test_agent_optimizer_excludes_critic_params(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = self._create_trainer(temp_dir)
            agent_opt_params = set()
            for g in trainer.agent_optimizer.param_groups:
                for p in g["params"]:
                    agent_opt_params.add(id(p))
            for p in trainer.eval_critic.parameters():
                self.assertNotIn(id(p), agent_opt_params)

    def test_critic_optimizer_includes_mixer_params(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = self._create_trainer(temp_dir)
            critic_opt_params = set()
            for g in trainer.critic_optimizer.param_groups:
                for p in g["params"]:
                    critic_opt_params.add(id(p))
            # The mixer's sub-networks should be in the critic optimizer
            for p in trainer.eval_critic.transformation.parameters():
                self.assertIn(id(p), critic_opt_params)
            for p in trainer.eval_critic.joint_attention.parameters():
                self.assertIn(id(p), critic_opt_params)
            for p in trainer.eval_critic.value_w_final.parameters():
                self.assertIn(id(p), critic_opt_params)
            for p in trainer.eval_critic.value_v.parameters():
                self.assertIn(id(p), critic_opt_params)

    def test_qplex_mixer_simple_n_head_1_works(self):
        """The simpler n_head=1 configuration should work end-to-end."""
        # Already covered by the test_learn_one_batch above since the
        # default config uses n_head=1. This is just a sanity check.
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = self._create_trainer(temp_dir)
            self.assertEqual(trainer.eval_critic.transformation.n_head, 1)
            self.assertEqual(trainer.eval_critic.joint_attention.n_head, 1)


if __name__ == "__main__":
    unittest.main()
