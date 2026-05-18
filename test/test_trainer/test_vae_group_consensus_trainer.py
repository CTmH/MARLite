import unittest
import os
import yaml
from copy import deepcopy
import tempfile
import torch

from marlite.trainer import TrainerConfig, REGISTERED_TRAINERS


class TestGroupConsensusTrainer(unittest.TestCase):
    def setUp(self):
        self.config_path = "test/config/vae_group_consensus_default.yaml"
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

    def test_torch_compile(self):
        self.config_path = "test/config/vae_group_consensus_default.yaml"
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

    def test_distributed_data_parallel(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping multi-GPU test")

        gpu_count = torch.cuda.device_count()
        if gpu_count < 2:
            self.skipTest(f"Need at least 2 GPUs for multi-GPU test, found {gpu_count}")

        self.config_path = "test/config/vae_group_consensus_default.yaml"
        with open(self.config_path, "r") as file:
            self.config = yaml.safe_load(file)
        self.config["trainer"]["train_args"]["epochs"] = 2
        self.config["rollout"]["n_episodes"] = 2
        self.config["rollout"]["n_eval_episodes"] = 2
        self.config["rollout"]["episode_limit"] = 2
        self.config["replay_buffer"]["capacity"] = 2

        device_list = [f"cuda:{i}" for i in range(gpu_count)]
        self.config["trainer"]["train_device"] = device_list
        self.trainer_config = TrainerConfig(self.config)

        with tempfile.TemporaryDirectory() as temp_dir:
            self.trainer = self.trainer_config.create_trainer()
            self.trainer.workdir = temp_dir
            self.trainer.logdir = os.path.join(self.trainer.workdir, "logs")
            self.trainer.checkpointdir = os.path.join(
                self.trainer.workdir, "checkpoints"
            )

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
            self.trainer.learn(sample_size=32, batch_size=8, times=2)
            self.trainer.update_target_model_params()

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
