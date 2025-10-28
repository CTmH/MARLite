import unittest
import os
import yaml
from copy import deepcopy
import tempfile
import torch

from marlite.trainer import TrainerConfig

class TestMsgAggrQMIXTrainer(unittest.TestCase):
    def setUp(self):
        self.config_path = 'test/config/msg_aggr_default.yaml'
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.config['trainer_config']['train_args']['epochs'] = 2
        self.config['rollout_config']['n_episodes'] = 2
        self.config['rollout_config']['n_eval_episodes'] = 2
        self.config['rollout_config']['episode_limit'] = 2
        self.config['replaybuffer_config']['capacity'] = 2
        self.trainer_config = TrainerConfig(self.config)

    def test_collect_experience(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, 'logs')
            self.trainer.checkpointdir = os.path.join(self.trainer.workdir, 'checkpoints')
            n_episodes = 4
            self.trainer.collect_experience(0.9)
            self.assertNotEqual(len(self.trainer.replaybuffer.buffer), 0)

    def test_learn(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, 'logs')
            self.trainer.checkpointdir = os.path.join(self.trainer.workdir, 'checkpoints')
            origin_critic_params = deepcopy(self.trainer.target_critic.state_dict())
            origin_agent_group_params = deepcopy(self.trainer.target_agent_group.get_agent_group_params())
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=2)
            self.trainer.update_target_model_params()
            critic_params = self.trainer.target_critic.state_dict()
            agent_group_params = self.trainer.target_agent_group.get_agent_group_params()

            # Check if critic parameters have changed
            for (w_name, w1), w2 in zip(critic_params.items(), origin_critic_params.values()):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))

            # Check if agent_group parameters have changed
            for param_type in ['encoder', 'feature_extractor', 'decoder', 'aggr_model']:
                if param_type in agent_group_params:
                    for model_name, params in agent_group_params[param_type].items():
                        if model_name in origin_agent_group_params[param_type]:
                            orig_params = origin_agent_group_params[param_type][model_name]
                            if isinstance(params, torch.Tensor):
                                self.assertFalse(torch.equal(params, orig_params))
                            else:
                                for param_name, param in params.items():
                                    if param.requires_grad:
                                        self.assertFalse(torch.equal(param, orig_params[param_name]),
                                                    f"{param_type} {model_name} {param_name} did not change")

    def test_save_load_checkpoint(self):
        checkpoint = 'test_checkpoint'
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, 'logs')
            self.trainer.checkpointdir = os.path.join(self.trainer.workdir, 'checkpoints')
            self.trainer.save_current_model(checkpoint)
            self.trainer.load_checkpoint(checkpoint)

    def test_train(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, 'logs')
            self.trainer.checkpointdir = os.path.join(self.trainer.workdir, 'checkpoints')
            reward, _, _ = self.trainer.evaluate()
            best_reward, _ = self.trainer.train(epochs=2, target_reward=5)

class TestMsgAggrSMACQMIXTrainer(unittest.TestCase):
    def setUp(self):
        self.config_path = 'test/config/msg_aggr_smac.yaml'
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.config['trainer_config']['train_args']['epochs'] = 2
        self.config['rollout_config']['n_episodes'] = 2
        self.config['rollout_config']['n_eval_episodes'] = 2
        self.config['rollout_config']['episode_limit'] = 2
        self.config['replaybuffer_config']['capacity'] = 2
        self.trainer_config = TrainerConfig(self.config)

    def test_collect_experience(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, 'logs')
            self.trainer.checkpointdir = os.path.join(self.trainer.workdir, 'checkpoints')
            n_episodes = 4
            self.trainer.collect_experience(0.9)
            self.assertNotEqual(len(self.trainer.replaybuffer.buffer), 0)

    def test_learn(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, 'logs')
            self.trainer.checkpointdir = os.path.join(self.trainer.workdir, 'checkpoints')
            origin_critic_params = deepcopy(self.trainer.target_critic.state_dict())
            origin_agent_group_params = deepcopy(self.trainer.target_agent_group.get_agent_group_params())
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=2)
            self.trainer.update_target_model_params()
            critic_params = self.trainer.target_critic.state_dict()
            agent_group_params = self.trainer.target_agent_group.get_agent_group_params()

            # Check if critic parameters have changed
            for (w_name, w1), w2 in zip(critic_params.items(), origin_critic_params.values()):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))

            # Check if agent_group parameters have changed
            for param_type in ['encoder', 'feature_extractor', 'decoder', 'aggr_model']:
                if param_type in agent_group_params:
                    for model_name, params in agent_group_params[param_type].items():
                        if model_name in origin_agent_group_params[param_type]:
                            orig_params = origin_agent_group_params[param_type][model_name]
                            if isinstance(params, torch.Tensor):
                                self.assertFalse(torch.equal(params, orig_params))
                            else:
                                for param_name, param in params.items():
                                    if param.requires_grad:
                                        self.assertFalse(torch.equal(param, orig_params[param_name]),
                                                    f"{param_type} {model_name} {param_name} did not change")

    def test_save_load_checkpoint(self):
        checkpoint = 'test_checkpoint'
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, 'logs')
            self.trainer.checkpointdir = os.path.join(self.trainer.workdir, 'checkpoints')
            self.trainer.save_current_model(checkpoint)
            self.trainer.load_checkpoint(checkpoint)

    def test_train(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, 'logs')
            self.trainer.checkpointdir = os.path.join(self.trainer.workdir, 'checkpoints')
            reward, _, _ = self.trainer.evaluate()
            best_reward, _ = self.trainer.train(epochs=2, target_reward=5)

    def test_data_parallel(self):
        self.config_path = 'test/config/msg_aggr_smac.yaml'
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.config['trainer_config']['train_args']['epochs'] = 2
        self.config['rollout_config']['n_episodes'] = 2
        self.config['rollout_config']['n_eval_episodes'] = 2
        self.config['rollout_config']['episode_limit'] = 2
        self.config['replaybuffer_config']['capacity'] = 2
        self.config['trainer_config']['use_data_parallel'] = True
        self.config['trainer_config']['train_device'] = 'cuda'
        self.trainer_config = TrainerConfig(self.config)
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, 'logs')
            self.trainer.checkpointdir = os.path.join(self.trainer.workdir, 'checkpoints')
            origin_critic_params = deepcopy(self.trainer.eval_critic.state_dict())
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=1)
            self.trainer.update_target_model_params()
            critic_params = self.trainer.eval_critic.state_dict()

            for w1, w2 in zip(critic_params.values(), origin_critic_params.values()):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))

    def test_torch_compile(self):
        self.config_path = 'test/config/msg_aggr_smac.yaml'
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.config['trainer_config']['train_args']['epochs'] = 2
        self.config['rollout_config']['n_episodes'] = 2
        self.config['rollout_config']['n_eval_episodes'] = 2
        self.config['rollout_config']['episode_limit'] = 2
        self.config['replaybuffer_config']['capacity'] = 2
        self.config['trainer_config']['compile_models'] = True
        self.trainer_config = TrainerConfig(self.config)
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, 'logs')
            self.trainer.checkpointdir = os.path.join(self.trainer.workdir, 'checkpoints')
            origin_critic_params = deepcopy(self.trainer.target_critic.state_dict())
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=1)
            self.trainer.update_target_model_params()
            critic_params = self.trainer.target_critic.state_dict()

            for w1, w2 in zip(critic_params.values(), origin_critic_params.values()):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))

class TestSeqMsgAggrSMACQMIXTrainer(unittest.TestCase):
    def setUp(self):
        self.config_path = 'test/config/seq_msg_aggr_smac.yaml'
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.config['trainer_config']['train_args']['epochs'] = 2
        self.config['rollout_config']['n_episodes'] = 2
        self.config['rollout_config']['n_eval_episodes'] = 2
        self.config['rollout_config']['episode_limit'] = 2
        self.config['replaybuffer_config']['capacity'] = 2
        self.trainer_config = TrainerConfig(self.config)

    def test_collect_experience(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, 'logs')
            self.trainer.checkpointdir = os.path.join(self.trainer.workdir, 'checkpoints')
            n_episodes = 4
            self.trainer.collect_experience(0.9)
            self.assertNotEqual(len(self.trainer.replaybuffer.buffer), 0)

    def test_learn(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, 'logs')
            self.trainer.checkpointdir = os.path.join(self.trainer.workdir, 'checkpoints')
            origin_critic_params = deepcopy(self.trainer.target_critic.state_dict())
            origin_agent_group_params = deepcopy(self.trainer.target_agent_group.get_agent_group_params())
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=2)
            self.trainer.update_target_model_params()
            critic_params = self.trainer.target_critic.state_dict()
            agent_group_params = self.trainer.target_agent_group.get_agent_group_params()

            # Check if critic parameters have changed
            for (w_name, w1), w2 in zip(critic_params.items(), origin_critic_params.values()):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))

            # Check if agent_group parameters have changed
            for param_type in ['encoder', 'feature_extractor', 'decoder', 'aggr_model']:
                if param_type in agent_group_params:
                    for model_name, params in agent_group_params[param_type].items():
                        if model_name in origin_agent_group_params[param_type]:
                            orig_params = origin_agent_group_params[param_type][model_name]
                            if isinstance(params, torch.Tensor):
                                self.assertFalse(torch.equal(params, orig_params))
                            else:
                                for param_name, param in params.items():
                                    if param.requires_grad:
                                        self.assertFalse(torch.equal(param, orig_params[param_name]),
                                                    f"{param_type} {model_name} {param_name} did not change")

    def test_save_load_checkpoint(self):
        checkpoint = 'test_checkpoint'
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, 'logs')
            self.trainer.checkpointdir = os.path.join(self.trainer.workdir, 'checkpoints')
            self.trainer.save_current_model(checkpoint)
            self.trainer.load_checkpoint(checkpoint)

    def test_train(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, 'logs')
            self.trainer.checkpointdir = os.path.join(self.trainer.workdir, 'checkpoints')
            reward, _, _ = self.trainer.evaluate()
            best_reward, _ = self.trainer.train(epochs=2, target_reward=5)

    def test_data_parallel(self):
        self.config_path = 'test/config/msg_aggr_smac.yaml'
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.config['trainer_config']['train_args']['epochs'] = 2
        self.config['rollout_config']['n_episodes'] = 2
        self.config['rollout_config']['n_eval_episodes'] = 2
        self.config['rollout_config']['episode_limit'] = 2
        self.config['replaybuffer_config']['capacity'] = 2
        self.config['trainer_config']['use_data_parallel'] = True
        self.config['trainer_config']['train_device'] = 'cuda'
        self.trainer_config = TrainerConfig(self.config)
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, 'logs')
            self.trainer.checkpointdir = os.path.join(self.trainer.workdir, 'checkpoints')
            origin_critic_params = deepcopy(self.trainer.eval_critic.state_dict())
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=1)
            self.trainer.update_target_model_params()
            critic_params = self.trainer.eval_critic.state_dict()

            for w1, w2 in zip(critic_params.values(), origin_critic_params.values()):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))

    def test_torch_compile(self):
        self.config_path = 'test/config/msg_aggr_smac.yaml'
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.config['trainer_config']['train_args']['epochs'] = 2
        self.config['rollout_config']['n_episodes'] = 2
        self.config['rollout_config']['n_eval_episodes'] = 2
        self.config['rollout_config']['episode_limit'] = 2
        self.config['replaybuffer_config']['capacity'] = 2
        self.config['trainer_config']['compile_models'] = True
        self.trainer_config = TrainerConfig(self.config)
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, 'logs')
            self.trainer.checkpointdir = os.path.join(self.trainer.workdir, 'checkpoints')
            origin_critic_params = deepcopy(self.trainer.target_critic.state_dict())
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=1)
            self.trainer.update_target_model_params()
            critic_params = self.trainer.target_critic.state_dict()

            for w1, w2 in zip(critic_params.values(), origin_critic_params.values()):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))

class TestProbSeqMsgAggrSMACQMIXTrainer(unittest.TestCase):
    def setUp(self):
        self.config_path = 'test/config/prob_seq_msg_aggr_smac.yaml'
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.config['trainer_config']['train_args']['epochs'] = 2
        self.config['rollout_config']['n_episodes'] = 2
        self.config['rollout_config']['n_eval_episodes'] = 2
        self.config['rollout_config']['episode_limit'] = 2
        self.config['replaybuffer_config']['capacity'] = 2
        self.trainer_config = TrainerConfig(self.config)

    def test_collect_experience(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, 'logs')
            self.trainer.checkpointdir = os.path.join(self.trainer.workdir, 'checkpoints')
            n_episodes = 4
            self.trainer.collect_experience(0.9)
            self.assertNotEqual(len(self.trainer.replaybuffer.buffer), 0)

    def test_learn(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, 'logs')
            self.trainer.checkpointdir = os.path.join(self.trainer.workdir, 'checkpoints')
            origin_critic_params = deepcopy(self.trainer.target_critic.state_dict())
            origin_agent_group_params = deepcopy(self.trainer.target_agent_group.get_agent_group_params())
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=2)
            self.trainer.update_target_model_params()
            critic_params = self.trainer.target_critic.state_dict()
            agent_group_params = self.trainer.target_agent_group.get_agent_group_params()

            # Check if critic parameters have changed
            for (w_name, w1), w2 in zip(critic_params.items(), origin_critic_params.values()):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))

            # Check if agent_group parameters have changed
            for param_type in ['encoder', 'feature_extractor', 'decoder', 'aggr_model']:
                if param_type in agent_group_params:
                    for model_name, params in agent_group_params[param_type].items():
                        if model_name in origin_agent_group_params[param_type]:
                            orig_params = origin_agent_group_params[param_type][model_name]
                            if isinstance(params, torch.Tensor):
                                self.assertFalse(torch.equal(params, orig_params))
                            else:
                                for param_name, param in params.items():
                                    if param.requires_grad:
                                        self.assertFalse(torch.equal(param, orig_params[param_name]),
                                                    f"{param_type} {model_name} {param_name} did not change")

    def test_save_load_checkpoint(self):
        checkpoint = 'test_checkpoint'
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, 'logs')
            self.trainer.checkpointdir = os.path.join(self.trainer.workdir, 'checkpoints')
            self.trainer.save_current_model(checkpoint)
            self.trainer.load_checkpoint(checkpoint)

    def test_train(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, 'logs')
            self.trainer.checkpointdir = os.path.join(self.trainer.workdir, 'checkpoints')
            reward, _, _ = self.trainer.evaluate()
            best_reward, _ = self.trainer.train(epochs=2, target_reward=5)

    def test_data_parallel(self):
        self.config_path = 'test/config/prob_seq_msg_aggr_smac.yaml'
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.config['trainer_config']['train_args']['epochs'] = 2
        self.config['rollout_config']['n_episodes'] = 2
        self.config['rollout_config']['n_eval_episodes'] = 2
        self.config['rollout_config']['episode_limit'] = 2
        self.config['replaybuffer_config']['capacity'] = 2
        self.config['trainer_config']['use_data_parallel'] = True
        self.config['trainer_config']['train_device'] = 'cuda'
        self.trainer_config = TrainerConfig(self.config)
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, 'logs')
            self.trainer.checkpointdir = os.path.join(self.trainer.workdir, 'checkpoints')
            origin_critic_params = deepcopy(self.trainer.eval_critic.state_dict())
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=1)
            self.trainer.update_target_model_params()
            critic_params = self.trainer.eval_critic.state_dict()

            for w1, w2 in zip(critic_params.values(), origin_critic_params.values()):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))

    def test_torch_compile(self):
        self.config_path = 'test/config/prob_seq_msg_aggr_smac.yaml'
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.config['trainer_config']['train_args']['epochs'] = 2
        self.config['rollout_config']['n_episodes'] = 2
        self.config['rollout_config']['n_eval_episodes'] = 2
        self.config['rollout_config']['episode_limit'] = 2
        self.config['replaybuffer_config']['capacity'] = 2
        self.config['trainer_config']['compile_models'] = True
        self.trainer_config = TrainerConfig(self.config)
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, 'logs')
            self.trainer.checkpointdir = os.path.join(self.trainer.workdir, 'checkpoints')
            origin_critic_params = deepcopy(self.trainer.target_critic.state_dict())
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=1)
            self.trainer.update_target_model_params()
            critic_params = self.trainer.target_critic.state_dict()

            for w1, w2 in zip(critic_params.values(), origin_critic_params.values()):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))

class TestDualPathObsMsgAggrSMACQMIXTrainer(unittest.TestCase):
    def setUp(self):
        self.config_path = 'test/config/dual_path_obs_msg_aggr_smac.yaml'
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.config['trainer_config']['train_args']['epochs'] = 2
        self.config['rollout_config']['n_episodes'] = 2
        self.config['rollout_config']['n_eval_episodes'] = 2
        self.config['rollout_config']['episode_limit'] = 2
        self.config['replaybuffer_config']['capacity'] = 2
        self.trainer_config = TrainerConfig(self.config)

    def test_collect_experience(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, 'logs')
            self.trainer.checkpointdir = os.path.join(self.trainer.workdir, 'checkpoints')
            n_episodes = 4
            self.trainer.collect_experience(0.9)
            self.assertNotEqual(len(self.trainer.replaybuffer.buffer), 0)

    def test_learn(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, 'logs')
            self.trainer.checkpointdir = os.path.join(self.trainer.workdir, 'checkpoints')
            origin_critic_params = deepcopy(self.trainer.target_critic.state_dict())
            origin_agent_group_params = deepcopy(self.trainer.target_agent_group.get_agent_group_params())
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=2)
            self.trainer.update_target_model_params()
            critic_params = self.trainer.target_critic.state_dict()
            agent_group_params = self.trainer.target_agent_group.get_agent_group_params()

            # Check if critic parameters have changed
            for (w_name, w1), w2 in zip(critic_params.items(), origin_critic_params.values()):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))

            # Check if agent_group parameters have changed
            for param_type in ['encoder', 'feature_extractor', 'decoder', 'aggr_model']:
                if param_type in agent_group_params:
                    for model_name, params in agent_group_params[param_type].items():
                        if model_name in origin_agent_group_params[param_type]:
                            orig_params = origin_agent_group_params[param_type][model_name]
                            if isinstance(params, torch.Tensor):
                                self.assertFalse(torch.equal(params, orig_params))
                            else:
                                for param_name, param in params.items():
                                    if param.requires_grad:
                                        self.assertFalse(torch.equal(param, orig_params[param_name]),
                                                    f"{param_type} {model_name} {param_name} did not change")

    def test_save_load_checkpoint(self):
        checkpoint = 'test_checkpoint'
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, 'logs')
            self.trainer.checkpointdir = os.path.join(self.trainer.workdir, 'checkpoints')
            self.trainer.save_current_model(checkpoint)
            self.trainer.load_checkpoint(checkpoint)

    def test_train(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, 'logs')
            self.trainer.checkpointdir = os.path.join(self.trainer.workdir, 'checkpoints')
            reward, _, _ = self.trainer.evaluate()
            best_reward, _ = self.trainer.train(epochs=2, target_reward=5)

    def test_data_parallel(self):
        self.config_path = 'test/config/prob_seq_msg_aggr_smac.yaml'
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.config['trainer_config']['train_args']['epochs'] = 2
        self.config['rollout_config']['n_episodes'] = 2
        self.config['rollout_config']['n_eval_episodes'] = 2
        self.config['rollout_config']['episode_limit'] = 2
        self.config['replaybuffer_config']['capacity'] = 2
        self.config['trainer_config']['use_data_parallel'] = True
        self.config['trainer_config']['train_device'] = 'cuda'
        self.trainer_config = TrainerConfig(self.config)
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, 'logs')
            self.trainer.checkpointdir = os.path.join(self.trainer.workdir, 'checkpoints')
            origin_critic_params = deepcopy(self.trainer.eval_critic.state_dict())
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=1)
            self.trainer.update_target_model_params()
            critic_params = self.trainer.eval_critic.state_dict()

            for w1, w2 in zip(critic_params.values(), origin_critic_params.values()):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))

    def test_torch_compile(self):
        self.config_path = 'test/config/prob_seq_msg_aggr_smac.yaml'
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.config['trainer_config']['train_args']['epochs'] = 2
        self.config['rollout_config']['n_episodes'] = 2
        self.config['rollout_config']['n_eval_episodes'] = 2
        self.config['rollout_config']['episode_limit'] = 2
        self.config['replaybuffer_config']['capacity'] = 2
        self.config['trainer_config']['compile_models'] = True
        self.trainer_config = TrainerConfig(self.config)
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, 'logs')
            self.trainer.checkpointdir = os.path.join(self.trainer.workdir, 'checkpoints')
            origin_critic_params = deepcopy(self.trainer.target_critic.state_dict())
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=1)
            self.trainer.update_target_model_params()
            critic_params = self.trainer.target_critic.state_dict()

            for w1, w2 in zip(critic_params.values(), origin_critic_params.values()):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))

if __name__ == '__main__':
    unittest.main()