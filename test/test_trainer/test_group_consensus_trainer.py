import unittest
import yaml
from copy import deepcopy
import tempfile
import torch

from marlite.trainer import TrainerConfig, REGISTERED_TRAINERS


class TestGroupConsensusTrainer(unittest.TestCase):
    yaml_config = """
agent_group:
  type: "GroupConsensusQMIX"
  agent_list:
    agent_0: model1
    agent_1: model1
    agent_2: model1
  models:
    model1:
      feature_extractor:
        model_type: "Custom"
        layers:
        - type: Linear
          in_features: 18
          out_features: 32
      group_estimate_feature_extractor:
        model_type: "Custom"
        layers:
        - type: Linear
          in_features: 18
          out_features: 16
      encoder:
        model_type: "Custom"
        layers:
        - type: Linear
          in_features: 32
          out_features: 8
      decoder:
        model_type: "Custom"
        layers:
        - type: Linear
          in_features: 16
          out_features: 5
  group_builder:
    type: "Fixed"
    group_ids: [0, 0, 1]
  deterministic_eval: true
  enable_rl_grad_to_group_estimate: false
  optimizer:
    type: "Adam"
    lr: 0.0005
    weight_decay: 0.0001

environment:
  module_name: "mpe2"
  env_name: "simple_spread_v3"

critic:
  type: "GroupConsensusMixer"
  feature_extractor:
    model_type: "Custom"
    layers:
    - type: Linear
      in_features: 54
      out_features: 32
  consensus_processor:
    model_type: "HyperNetwork"
    cond_dim: 32
    layer_dims: [24, 32, 32]
    cond_hidden_dim: 64
  model:
    model_type: "QMixModel"
    state_shape: 32
    input_dim: 3
    qmix_hidden_dim: 32
  num_agents: 3
  group_latent_dim: 8
  deterministic_eval: true
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
  episode_limit: 2
  device: "cpu"

replay_buffer:
  type: "Normal"
  capacity: 2
  traj_len: 7

analyzer:
  type: "default"

trainer:
  type: "GroupConsensusQMIX"
  gamma: 0.95
  eval_epsilon: 0.01
  workdir: "./test/results/replace_by_tempfile"
  train_device: "cpu"
  kl_divergence_weight: 0.005
  warmup_epochs: 0

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
            n_episodes = 4
            trainer.collect_experience(0.9)
            self.assertNotEqual(len(trainer.replaybuffer.buffer), 0)

    def test_learn(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = self._create_trainer(temp_dir)
            origin_critic_params = deepcopy(trainer.target_critic.state_dict())
            origin_agent_group_params = deepcopy(
                trainer.target_agent_group.state_dict()
            )
            trainer.collect_experience(0.9)
            trainer.learn(sample_size=32, batch_size=8, times=1)
            trainer._update_target_after_batch()
            critic_params = trainer.target_critic.state_dict()
            agent_group_params = trainer.target_agent_group.state_dict()

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
            trainer = self._create_trainer(temp_dir)
            trainer.save_current_model(checkpoint)
            trainer.load_checkpoint(checkpoint)

    def test_train(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = self._create_trainer(temp_dir)
            result = trainer.evaluate()
            best_metrics = trainer.train(epochs=2, target_first_metric=5)

    def test_torch_compile(self):
        config = deepcopy(self.config)
        config["trainer"]["compile_models"] = True
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = self._create_trainer(temp_dir, config=config)
            origin_critic_params = deepcopy(trainer.target_critic.state_dict())
            trainer.collect_experience(0.9)
            trainer.learn(sample_size=32, batch_size=8, times=1)
            trainer._update_target_after_batch()
            critic_params = trainer.target_critic.state_dict()

            for w1, w2 in zip(critic_params.values(), origin_critic_params.values()):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))


if __name__ == "__main__":
    unittest.main()
