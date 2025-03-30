import sys    
print("In module products sys.path[0], __package__ ==", sys.path[0], __package__)
import unittest
import torch
import yaml
import numpy as np
from unittest.mock import MagicMock
from pettingzoo.mpe import simple_spread_v3
from torch import Tensor, stack

from src.algorithm.agents import QMIXAgentGroup
from src.algorithm.agents.agent_group_config import AgentGroupConfig
from src.algorithm.model import ModelConfig
from src.util.optimizer_config import OptimizerConfig
from src.environment.env_config import EnvConfig


class TestGNNAgentGroup(unittest.TestCase):

    def setUp(self):
        # Agent group configuration
        config_path = 'test/config/gnn_default.yaml'
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.agent_group_config = AgentGroupConfig(**config['agent_group_config'])

        # Environment setup and model configuration
        self.env_config = EnvConfig(**config['env_config'])
        self.env = self.env_config.create_env()
        obs, _ = self.env.reset()
        key = self.env.agents[0]
        self.obs_shape = self.env.observation_space(key).shape
        self.obs_shape = self.obs_shape[0]
        self.action_space_shape = self.env.action_space(key).n
        
        # Initialize QMIXAgents
        self.agent_group = self.agent_group_config.get_agent_group()

        observations = {agent: [] for agent in self.env.agents}
        seq_length = 5
        for i in range(seq_length):
            actions = {agent: self.env.action_space(agent).sample() for agent in self.env.agents}
            obs, rewards, terminations, truncations, infos = self.env.step(actions)
            for agent in self.env.agents:
                observations[agent].append(obs[agent])
        self.observations = {key: np.array(value) for key, value in observations.items()}
        self.adj_matrix, self.edge_index = self.env.build_my_team_graph()
        self.adj_matrix, self.edge_index = np.expand_dims(self.adj_matrix, axis=0), np.expand_dims(self.edge_index, axis=0)
        
    def test_foward(self):
        obs = [self.observations[ag] for ag in self.agent_group.agent_model_dict.keys()]
        obs = np.stack(obs)
        obs = np.expand_dims(obs, axis=0)
        obs = torch.Tensor(obs)
        # Test get_q_values method in evaluation mode
        q_values = self.agent_group.forward(observations=obs, edge_index=self.edge_index)
        q_values = q_values.detach().cpu().numpy().squeeze()
        self.assertEqual(q_values.shape, (len(self.env.agents), self.action_space_shape))
        
        # Test get_q_values method in training mode
        self.agent_group.train()
        q_values = self.agent_group.forward(observations=obs, edge_index=self.edge_index)
        q_values = q_values.detach().cpu().numpy().squeeze()
        self.assertEqual(q_values.shape, (len(self.env.agents), self.action_space_shape))

    def test_act(self):
        # Test act method with epsilon = 0 (greedy policy)
        edge_index = self.edge_index[0]
        actions = self.agent_group.act(self.observations, edge_index, self.env.action_spaces, epsilon=0)
        self.assertEqual(len(actions), len(self.env.agents))

        # Test act method with epsilon = 1 (random policy)
        actions = self.agent_group.act(self.observations, edge_index, self.env.action_spaces, epsilon=1)
        self.assertEqual(len(actions), len(self.env.agents))

        # Test act method with epsilon = 0.5
        actions = self.agent_group.act(self.observations, edge_index, self.env.action_spaces, epsilon=0.5)
        self.assertEqual(len(actions), len(self.env.agents))

    def test_eval(self):
        self.agent_group.eval()
        # Check if the agent group is in evaluation mode
        for (model_name, model), (_, fe) in zip(self.agent_group.models.items(), self.agent_group.feature_extractors.items()):
            self.assertFalse(model.training)
            self.assertFalse(fe.training)

    def test_train(self):
        self.agent_group.train()
        # Check if the agent group is in training mode
        for (model_name, model), (_, fe) in zip(self.agent_group.models.items(), self.agent_group.feature_extractors.items()):
            self.assertTrue(model.training)
            self.assertTrue(fe.training)
        
if __name__ == '__main__':
    unittest.main()
