import unittest
import yaml
import tempfile
import torch
from marlite.trainer import TrainerConfig


QTRAN_CFG = """
agent_group:
  type: "QTRAN"
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
  type: "Qtransform"
  action_dim: 5
  base_model:
    model_type: "Custom"
    layers:
    - type: "Linear"
      in_features: 32
      out_features: 32
    - type: "ReLU"
    - type: "Linear"
      in_features: 32
      out_features: 5
  phi_net:
    model_type: "Custom"
    layers:
    - type: "Linear"
      in_features: 37
      out_features: 64
    - type: "ReLU"
    - type: "Linear"
      in_features: 64
      out_features: 32
  psi_net:
    model_type: "Custom"
    layers:
    - type: "Linear"
      in_features: 32
      out_features: 32
    - type: "ReLU"
    - type: "Linear"
      in_features: 32
      out_features: 32
  optimizer:
    type: "Adam"
    lr: 0.0005
    weight_decay: 0.0001

v_net:
  type: "StateValue"
  base_model:
    model_type: "Custom"
    layers:
    - type: "Linear"
      in_features: 32
      out_features: 1
  feature_extractor:
    model_type: "Custom"
    layers:
    - type: "Linear"
      in_features: 54
      out_features: 32
    - type: "ReLU"
  optimizer:
    type: "Adam"
    lr: 0.001

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
  type: "QTRAN"
  gamma: 0.95
  eval_epsilon: 0.01
  lambda_opt: 1.0
  lambda_nopt: 1.0
  is_optimal_mask_mode: true
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


class TestQTRANTrainer(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load(QTRAN_CFG)

    def _create_trainer(self, temp_dir):
        self.config["trainer"]["workdir"] = temp_dir
        trainer_config = TrainerConfig(self.config)
        return trainer_config.create_trainer()

    def test_create_trainer_has_three_modules(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = self._create_trainer(temp_dir)
            self.assertIsNotNone(trainer.eval_agent_group)
            self.assertIsNotNone(trainer.target_agent_group)
            self.assertIsNotNone(trainer.eval_critic)
            self.assertIsNotNone(trainer.target_critic)
            self.assertIsNotNone(trainer.eval_v_net)
            self.assertTrue(hasattr(trainer.eval_critic, "phi_net"))
            self.assertTrue(hasattr(trainer.eval_critic, "psi_net"))
            self.assertTrue(hasattr(trainer.target_critic, "phi_net"))
            self.assertTrue(hasattr(trainer.target_critic, "psi_net"))

    def test_is_optimal_mask_mode_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = self._create_trainer(temp_dir)
            self.assertTrue(trainer.is_optimal_mask_mode)

    def test_is_optimal_mask_mode_disabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = yaml.safe_load(QTRAN_CFG)
            cfg["trainer"]["is_optimal_mask_mode"] = False
            cfg["trainer"]["workdir"] = temp_dir
            trainer = TrainerConfig(cfg).create_trainer()
            self.assertFalse(trainer.is_optimal_mask_mode)

    def test_target_update_syncs_v_net(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = self._create_trainer(temp_dir)
            for p in trainer.eval_v_net.parameters():
                torch.nn.init.ones_(p)
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

    def test_create_trainer_has_three_optimizers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = self._create_trainer(temp_dir)
            self.assertIsNotNone(trainer.agent_optimizer)
            self.assertIsNotNone(trainer.critic_optimizer)
            self.assertIsNotNone(trainer.v_optimizer)

    def test_critic_optimizer_excludes_v_params(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = self._create_trainer(temp_dir)
            critic_opt_params = set()
            for g in trainer.critic_optimizer.param_groups:
                for p in g["params"]:
                    critic_opt_params.add(id(p))
            for p in trainer.eval_v_net.parameters():
                self.assertNotIn(id(p), critic_opt_params)

    def test_v_optimizer_only_updates_v_params(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = self._create_trainer(temp_dir)
            v_opt_params = set()
            for g in trainer.v_optimizer.param_groups:
                for p in g["params"]:
                    v_opt_params.add(id(p))
            for p in trainer.eval_critic.parameters():
                self.assertNotIn(id(p), v_opt_params)
            for p in trainer.eval_agent_group.parameters():
                self.assertNotIn(id(p), v_opt_params)

    def test_critic_optimizer_includes_encoder_params(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = self._create_trainer(temp_dir)
            critic_opt_params = set()
            for g in trainer.critic_optimizer.param_groups:
                for p in g["params"]:
                    critic_opt_params.add(id(p))
            for p in trainer.eval_critic.phi_net.parameters():
                self.assertIn(id(p), critic_opt_params)
            for p in trainer.eval_critic.psi_net.parameters():
                self.assertIn(id(p), critic_opt_params)
            for p in trainer.eval_critic.base_model.parameters():
                self.assertIn(id(p), critic_opt_params)

    def test_target_update_syncs_agent_and_critic(self):
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
                "v_net": {k: v.clone() for k, v in trainer.eval_v_net.state_dict().items()},
            }

            for p in trainer.eval_agent_group.parameters():
                torch.nn.init.zeros_(p)
            for p in trainer.eval_critic.parameters():
                torch.nn.init.zeros_(p)
            for p in trainer.eval_v_net.parameters():
                torch.nn.init.zeros_(p)

            trainer.load_checkpoint("ckpt0")

            for k, v in params_before["agent"].items():
                self.assertTrue(torch.equal(v, trainer.eval_agent_group.state_dict()[k]))
            for k, v in params_before["critic"].items():
                self.assertTrue(torch.equal(v, trainer.eval_critic.state_dict()[k]))
            for k, v in params_before["v_net"].items():
                self.assertTrue(torch.equal(v, trainer.eval_v_net.state_dict()[k]))


if __name__ == "__main__":
    unittest.main()
