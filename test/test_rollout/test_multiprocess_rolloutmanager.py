import unittest
import torch
import numpy as np
import yaml
from src.algorithm.agents import QMIXAgentGroup
from src.rollout.multiprocess_rolloutmanager import MultiProcessRolloutManager
from src.rollout.multiprocess_rolloutworker import MultiProcessRolloutWorker
from src.algorithm.model import ModelConfig
from src.environment.env_config import EnvConfig
from src.util.optimizer_config import OptimizerConfig
from src.algorithm.agents.agent_group_config import AgentGroupConfig
import torch.nn as nn
import torch.nn.init as init
def init_weights(m):
    if isinstance(m, nn.Linear):
        init.constant_(m.weight, 0.5)
        if m.bias is not None:
            init.constant_(m.bias, 0)

class TestRolloutManager(unittest.TestCase):

    def setUp(self):
        # Environment setup and model configuration
        self.env_config = {"module_name": "pettingzoo.mpe", "env_name": "simple_spread_v3"}
        self.env_config = EnvConfig(**self.env_config)
        # Agent group configuration
        config = """
            agent_group_config:
                type: "QMIX"
                agent_list:
                    agent_0: model1
                    agent_1: model1
                    agent_2: model1
                model_configs:
                    model1:
                        model_type: "RNN"
                        input_shape: 18
                        rnn_hidden_dim: 128
                        rnn_layers: 1
                        output_shape: 5
                        feature_extractor:
                            model_type: "Identity"
                optimizer:
                    type: "Adam"
                    lr: 0.0005
                    weight_decay: 0.0001
        """
        config = yaml.safe_load(config)
        self.agent_group_config = AgentGroupConfig(**config['agent_group_config'])
        
        # Initialize QMIXAgents models parameters
        self.agent_group = self.agent_group_config.get_agent_group()
        for model in self.agent_group.models.values():
            model.apply(init_weights)
        for fe in self.agent_group.feature_extractors.values():
            fe.apply(init_weights)
        self.agent_model_params, self.agent_fe_params = self.agent_group.get_agent_group_params()

        self.traj_len = 5
        self.n_episodes = 13
        self.episode_limit = 10
        self.n_workers = 1
        self.manager = MultiProcessRolloutManager(worker_class=MultiProcessRolloutWorker,
                                                  env_config=self.env_config,
                                                  agent_group_config=self.agent_group_config,
                                                  agent_model_params=self.agent_model_params,
                                                  agent_fe_params=self.agent_fe_params,
                                                  n_workers=self.n_workers,
                                                  n_episodes=self.n_episodes,
                                                  traj_len=self.traj_len,
                                                  episode_limit=self.episode_limit,
                                                  epsilon=0.9,
                                                  device='cpu')
'''
    def test_generate_episodes(self):
        episodes = self.manager.generate_episodes()
        self.assertEqual(len(episodes), self.n_episodes)
        for episode in episodes:
            self.assertEqual(len(episode['rewards']), self.episode_limit)
'''

if __name__ == '__main__':
    unittest.main()
