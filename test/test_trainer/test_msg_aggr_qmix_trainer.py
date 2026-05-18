import unittest
import os
import yaml
from copy import deepcopy
import tempfile
import torch

from marlite.trainer import TrainerConfig


class TestMsgAggrQMIXTrainer(unittest.TestCase):
    def setUp(self):
        self.config_path = "test/config/msg_aggr_default.yaml"
        with open(self.config_path, "r") as file:
            self.config = yaml.safe_load(file)
        self.config["trainer"]["train_args"]["epochs"] = 2
        self.config["rollout"]["n_episodes"] = 2
        self.config["rollout"]["n_eval_episodes"] = 2
        self.config["rollout"]["episode_limit"] = 2
        self.config["replay_buffer"]["capacity"] = 2
        self.trainer_config = TrainerConfig(self.config)

    def test_collect_experience(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, "logs")
            self.trainer.checkpointdir = os.path.join(
                self.trainer.workdir, "checkpoints"
            )
            n_episodes = 4
            self.trainer.collect_experience(0.9)
            self.assertNotEqual(len(self.trainer.replaybuffer.buffer), 0)

    def test_learn(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, "logs")
            self.trainer.checkpointdir = os.path.join(
                self.trainer.workdir, "checkpoints"
            )
            origin_critic_params = deepcopy(self.trainer.target_critic.state_dict())
            origin_agent_group_params = deepcopy(
                self.trainer.target_agent_group.state_dict()
            )
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=2)
            self.trainer.update_target_model_params()
            critic_params = self.trainer.target_critic.state_dict()
            agent_group_params = self.trainer.target_agent_group.state_dict()

            # Check if critic parameters have changed
            for (w_name, w1), w2 in zip(
                critic_params.items(), origin_critic_params.values()
            ):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))

            # Check if agent_group parameters have changed
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
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, "logs")
            self.trainer.checkpointdir = os.path.join(
                self.trainer.workdir, "checkpoints"
            )
            self.trainer.save_current_model(checkpoint)
            self.trainer.load_checkpoint(checkpoint)

    def test_train(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, "logs")
            self.trainer.checkpointdir = os.path.join(
                self.trainer.workdir, "checkpoints"
            )
            result = self.trainer.evaluate()
            best_metrics = self.trainer.train(epochs=2, target_first_metric=5)


class TestMsgAggrSMACQMIXTrainer(unittest.TestCase):
    def setUp(self):
        self.config_path = "test/config/msg_aggr_smac.yaml"
        with open(self.config_path, "r") as file:
            self.config = yaml.safe_load(file)
        self.config["trainer"]["train_args"]["epochs"] = 2
        self.config["rollout"]["n_episodes"] = 2
        self.config["rollout"]["n_eval_episodes"] = 2
        self.config["rollout"]["episode_limit"] = 2
        self.config["replay_buffer"]["capacity"] = 2
        self.trainer_config = TrainerConfig(self.config)

    def test_collect_experience(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, "logs")
            self.trainer.checkpointdir = os.path.join(
                self.trainer.workdir, "checkpoints"
            )
            n_episodes = 4
            self.trainer.collect_experience(0.9)
            self.assertNotEqual(len(self.trainer.replaybuffer.buffer), 0)

    def test_learn(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, "logs")
            self.trainer.checkpointdir = os.path.join(
                self.trainer.workdir, "checkpoints"
            )
            origin_critic_params = deepcopy(self.trainer.target_critic.state_dict())
            origin_agent_group_params = deepcopy(
                self.trainer.target_agent_group.state_dict()
            )
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=2)
            self.trainer.update_target_model_params()
            critic_params = self.trainer.target_critic.state_dict()
            agent_group_params = self.trainer.target_agent_group.state_dict()

            # Check if critic parameters have changed
            for (w_name, w1), w2 in zip(
                critic_params.items(), origin_critic_params.values()
            ):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))

            # Check if agent_group parameters have changed
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
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, "logs")
            self.trainer.checkpointdir = os.path.join(
                self.trainer.workdir, "checkpoints"
            )
            self.trainer.save_current_model(checkpoint)
            self.trainer.load_checkpoint(checkpoint)

    def test_train(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, "logs")
            self.trainer.checkpointdir = os.path.join(
                self.trainer.workdir, "checkpoints"
            )
            result = self.trainer.evaluate()
            best_metrics = self.trainer.train(epochs=2, target_first_metric=5)

    def test_distributed_data_parallel(self):
        """Test DistributedDataParallel training with proper DDP initialization."""
        self.config_path = "test/config/msg_aggr_smac.yaml"
        with open(self.config_path, "r") as file:
            self.config = yaml.safe_load(file)
        self.config["trainer"]["train_args"]["epochs"] = 2
        self.config["rollout"]["n_episodes"] = 2
        self.config["rollout"]["n_eval_episodes"] = 2
        self.config["rollout"]["episode_limit"] = 2
        self.config["replay_buffer"]["capacity"] = 2
        self.config["trainer"]["train_device"] = ["cuda:0"]
        self.trainer_config = TrainerConfig(self.config)

        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, "logs")
            self.trainer.checkpointdir = os.path.join(
                self.trainer.workdir, "checkpoints"
            )
            origin_critic_params = deepcopy(self.trainer.eval_critic.state_dict())
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=1)
            self.trainer.update_target_model_params()
            critic_params = self.trainer.eval_critic.state_dict()

            for w1, w2 in zip(critic_params.values(), origin_critic_params.values()):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))

    def test_torch_compile(self):
        self.config_path = "test/config/msg_aggr_smac.yaml"
        with open(self.config_path, "r") as file:
            self.config = yaml.safe_load(file)
        self.config["trainer"]["train_args"]["epochs"] = 2
        self.config["rollout"]["n_episodes"] = 2
        self.config["rollout"]["n_eval_episodes"] = 2
        self.config["rollout"]["episode_limit"] = 2
        self.config["replay_buffer"]["capacity"] = 2
        self.config["trainer"]["compile_models"] = True
        self.trainer_config = TrainerConfig(self.config)
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, "logs")
            self.trainer.checkpointdir = os.path.join(
                self.trainer.workdir, "checkpoints"
            )
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
        self.config_path = "test/config/seq_msg_aggr_smac.yaml"
        with open(self.config_path, "r") as file:
            self.config = yaml.safe_load(file)
        self.config["trainer"]["train_args"]["epochs"] = 2
        self.config["rollout"]["n_episodes"] = 2
        self.config["rollout"]["n_eval_episodes"] = 2
        self.config["rollout"]["episode_limit"] = 2
        self.config["replay_buffer"]["capacity"] = 2
        self.trainer_config = TrainerConfig(self.config)

    def test_collect_experience(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, "logs")
            self.trainer.checkpointdir = os.path.join(
                self.trainer.workdir, "checkpoints"
            )
            n_episodes = 4
            self.trainer.collect_experience(0.9)
            self.assertNotEqual(len(self.trainer.replaybuffer.buffer), 0)

    def test_learn(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, "logs")
            self.trainer.checkpointdir = os.path.join(
                self.trainer.workdir, "checkpoints"
            )
            origin_critic_params = deepcopy(self.trainer.target_critic.state_dict())
            origin_agent_group_params = deepcopy(
                self.trainer.target_agent_group.state_dict()
            )
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=2)
            self.trainer.update_target_model_params()
            critic_params = self.trainer.target_critic.state_dict()
            agent_group_params = self.trainer.target_agent_group.state_dict()

            # Check if critic parameters have changed
            for (w_name, w1), w2 in zip(
                critic_params.items(), origin_critic_params.values()
            ):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))

            # Check if agent_group parameters have changed
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
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, "logs")
            self.trainer.checkpointdir = os.path.join(
                self.trainer.workdir, "checkpoints"
            )
            self.trainer.save_current_model(checkpoint)
            self.trainer.load_checkpoint(checkpoint)

    def test_train(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, "logs")
            self.trainer.checkpointdir = os.path.join(
                self.trainer.workdir, "checkpoints"
            )
            result = self.trainer.evaluate()
            best_metrics = self.trainer.train(epochs=2, target_first_metric=5)

    def test_distributed_data_parallel(self):
        """Test DistributedDataParallel training with proper DDP initialization."""
        self.config_path = "test/config/msg_aggr_smac.yaml"
        with open(self.config_path, "r") as file:
            self.config = yaml.safe_load(file)
        self.config["trainer"]["train_args"]["epochs"] = 2
        self.config["rollout"]["n_episodes"] = 2
        self.config["rollout"]["n_eval_episodes"] = 2
        self.config["rollout"]["episode_limit"] = 2
        self.config["replay_buffer"]["capacity"] = 2
        self.config["trainer"]["train_device"] = ["cuda:0"]
        self.trainer_config = TrainerConfig(self.config)

        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, "logs")
            self.trainer.checkpointdir = os.path.join(
                self.trainer.workdir, "checkpoints"
            )
            origin_critic_params = deepcopy(self.trainer.eval_critic.state_dict())
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=1)
            self.trainer.update_target_model_params()
            critic_params = self.trainer.eval_critic.state_dict()

            for w1, w2 in zip(critic_params.values(), origin_critic_params.values()):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))

    def test_torch_compile(self):
        self.config_path = "test/config/msg_aggr_smac.yaml"
        with open(self.config_path, "r") as file:
            self.config = yaml.safe_load(file)
        self.config["trainer"]["train_args"]["epochs"] = 2
        self.config["rollout"]["n_episodes"] = 2
        self.config["rollout"]["n_eval_episodes"] = 2
        self.config["rollout"]["episode_limit"] = 2
        self.config["replay_buffer"]["capacity"] = 2
        self.config["trainer"]["compile_models"] = True
        self.trainer_config = TrainerConfig(self.config)
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, "logs")
            self.trainer.checkpointdir = os.path.join(
                self.trainer.workdir, "checkpoints"
            )
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
        self.config_path = "test/config/prob_seq_msg_aggr_smac.yaml"
        with open(self.config_path, "r") as file:
            self.config = yaml.safe_load(file)
        self.config["trainer"]["train_args"]["epochs"] = 2
        self.config["rollout"]["n_episodes"] = 2
        self.config["rollout"]["n_eval_episodes"] = 2
        self.config["rollout"]["episode_limit"] = 2
        self.config["replay_buffer"]["capacity"] = 2
        self.trainer_config = TrainerConfig(self.config)

    def test_deterministic_eval(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, "logs")
            self.trainer.checkpointdir = os.path.join(
                self.trainer.workdir, "checkpoints"
            )
            self.assertFalse(self.trainer.eval_agent_group.deterministic_eval)
            self.assertFalse(self.trainer.eval_critic.deterministic_eval)

    def test_collect_experience(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, "logs")
            self.trainer.checkpointdir = os.path.join(
                self.trainer.workdir, "checkpoints"
            )
            n_episodes = 4
            self.trainer.collect_experience(0.9)
            self.assertNotEqual(len(self.trainer.replaybuffer.buffer), 0)

    def test_learn(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, "logs")
            self.trainer.checkpointdir = os.path.join(
                self.trainer.workdir, "checkpoints"
            )
            origin_critic_params = deepcopy(self.trainer.target_critic.state_dict())
            origin_agent_group_params = deepcopy(
                self.trainer.target_agent_group.state_dict()
            )
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=2)
            self.trainer.update_target_model_params()
            critic_params = self.trainer.target_critic.state_dict()
            agent_group_params = self.trainer.target_agent_group.state_dict()

            # Check if critic parameters have changed
            for (w_name, w1), w2 in zip(
                critic_params.items(), origin_critic_params.values()
            ):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))

            # Check if agent_group parameters have changed
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
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, "logs")
            self.trainer.checkpointdir = os.path.join(
                self.trainer.workdir, "checkpoints"
            )
            self.trainer.save_current_model(checkpoint)
            self.trainer.load_checkpoint(checkpoint)

    def test_train(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, "logs")
            self.trainer.checkpointdir = os.path.join(
                self.trainer.workdir, "checkpoints"
            )
            result = self.trainer.evaluate()
            best_metrics = self.trainer.train(epochs=2, target_first_metric=5)

    def test_distributed_data_parallel(self):
        """Test DistributedDataParallel training with proper DDP initialization."""
        self.config_path = "test/config/msg_aggr_smac.yaml"
        with open(self.config_path, "r") as file:
            self.config = yaml.safe_load(file)
        self.config["trainer"]["train_args"]["epochs"] = 2
        self.config["rollout"]["n_episodes"] = 2
        self.config["rollout"]["n_eval_episodes"] = 2
        self.config["rollout"]["episode_limit"] = 2
        self.config["replay_buffer"]["capacity"] = 2
        self.config["trainer"]["train_device"] = ["cuda:0"]
        self.trainer_config = TrainerConfig(self.config)

        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, "logs")
            self.trainer.checkpointdir = os.path.join(
                self.trainer.workdir, "checkpoints"
            )
            origin_critic_params = deepcopy(self.trainer.eval_critic.state_dict())
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=1)
            self.trainer.update_target_model_params()
            critic_params = self.trainer.eval_critic.state_dict()

            for w1, w2 in zip(critic_params.values(), origin_critic_params.values()):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))

    def test_torch_compile(self):
        self.config_path = "test/config/prob_seq_msg_aggr_smac.yaml"
        with open(self.config_path, "r") as file:
            self.config = yaml.safe_load(file)
        self.config["trainer"]["train_args"]["epochs"] = 2
        self.config["rollout"]["n_episodes"] = 2
        self.config["rollout"]["n_eval_episodes"] = 2
        self.config["rollout"]["episode_limit"] = 2
        self.config["replay_buffer"]["capacity"] = 2
        self.config["trainer"]["compile_models"] = True
        self.trainer_config = TrainerConfig(self.config)
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, "logs")
            self.trainer.checkpointdir = os.path.join(
                self.trainer.workdir, "checkpoints"
            )
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
        self.config_path = "test/config/dual_path_obs_msg_aggr_smac.yaml"
        with open(self.config_path, "r") as file:
            self.config = yaml.safe_load(file)
        self.config["trainer"]["train_args"]["epochs"] = 2
        self.config["rollout"]["n_episodes"] = 2
        self.config["rollout"]["n_eval_episodes"] = 2
        self.config["rollout"]["episode_limit"] = 2
        self.config["replay_buffer"]["capacity"] = 2
        self.trainer_config = TrainerConfig(self.config)

    def test_collect_experience(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, "logs")
            self.trainer.checkpointdir = os.path.join(
                self.trainer.workdir, "checkpoints"
            )
            n_episodes = 4
            self.trainer.collect_experience(0.9)
            self.assertNotEqual(len(self.trainer.replaybuffer.buffer), 0)

    def test_learn(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, "logs")
            self.trainer.checkpointdir = os.path.join(
                self.trainer.workdir, "checkpoints"
            )
            origin_critic_params = deepcopy(self.trainer.target_critic.state_dict())
            origin_agent_group_params = deepcopy(
                self.trainer.target_agent_group.state_dict()
            )
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=2)
            self.trainer.update_target_model_params()
            critic_params = self.trainer.target_critic.state_dict()
            agent_group_params = self.trainer.target_agent_group.state_dict()

            # Check if critic parameters have changed
            for (w_name, w1), w2 in zip(
                critic_params.items(), origin_critic_params.values()
            ):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))

            # Check if agent_group parameters have changed
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
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, "logs")
            self.trainer.checkpointdir = os.path.join(
                self.trainer.workdir, "checkpoints"
            )
            self.trainer.save_current_model(checkpoint)
            self.trainer.load_checkpoint(checkpoint)

    def test_train(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, "logs")
            self.trainer.checkpointdir = os.path.join(
                self.trainer.workdir, "checkpoints"
            )
            result = self.trainer.evaluate()
            best_metrics = self.trainer.train(epochs=2, target_first_metric=5)

    def test_distributed_data_parallel(self):
        """Test DistributedDataParallel training with proper DDP initialization."""
        self.config_path = "test/config/msg_aggr_smac.yaml"
        with open(self.config_path, "r") as file:
            self.config = yaml.safe_load(file)
        self.config["trainer"]["train_args"]["epochs"] = 2
        self.config["rollout"]["n_episodes"] = 2
        self.config["rollout"]["n_eval_episodes"] = 2
        self.config["rollout"]["episode_limit"] = 2
        self.config["replay_buffer"]["capacity"] = 2
        self.config["trainer"]["train_device"] = ["cuda:0"]
        self.trainer_config = TrainerConfig(self.config)

        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, "logs")
            self.trainer.checkpointdir = os.path.join(
                self.trainer.workdir, "checkpoints"
            )
            origin_critic_params = deepcopy(self.trainer.eval_critic.state_dict())
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=1)
            self.trainer.update_target_model_params()
            critic_params = self.trainer.eval_critic.state_dict()

            for w1, w2 in zip(critic_params.values(), origin_critic_params.values()):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))

    def test_torch_compile(self):
        self.config_path = "test/config/dual_path_obs_msg_aggr_smac.yaml"
        with open(self.config_path, "r") as file:
            self.config = yaml.safe_load(file)
        self.config["trainer"]["train_args"]["epochs"] = 2
        self.config["rollout"]["n_episodes"] = 4
        self.config["rollout"]["n_eval_episodes"] = 2
        self.config["rollout"]["episode_limit"] = 16
        self.config["replay_buffer"]["capacity"] = 4
        self.config["trainer"]["compile_models"] = True
        self.trainer_config = TrainerConfig(self.config)
        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, "logs")
            self.trainer.checkpointdir = os.path.join(
                self.trainer.workdir, "checkpoints"
            )
            origin_critic_params = deepcopy(self.trainer.target_critic.state_dict())
            self.trainer.collect_experience(0.9)
            self.trainer.learn(sample_size=32, batch_size=8, times=1)
            self.trainer.update_target_model_params()
            critic_params = self.trainer.target_critic.state_dict()

            for w1, w2 in zip(critic_params.values(), origin_critic_params.values()):
                if w1.requires_grad:
                    self.assertFalse(torch.equal(w1, w2))


if __name__ == "__main__":
    unittest.main()
