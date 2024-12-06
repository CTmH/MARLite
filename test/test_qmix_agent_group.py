import sys    
print("In module products sys.path[0], __package__ ==", sys.path[0], __package__)
import unittest
import torch
import numpy as np
from unittest.mock import MagicMock
from pettingzoo.mpe import simple_spread_v3

from src.module.agents import QMIXAgentGroup
from src.module.model import ModelConfig

class TestQMIXAgentGroup(unittest.TestCase):

    def setUp(self):
        # Environment setup and model configuration
        self.env = simple_spread_v3.parallel_env(render_mode="human")
        obs, _ = self.env.reset()
        key = self.env.agents[0]
        self.obs_shape = self.env.observation_space(key).shape
        self.obs_shape = self.obs_shape[0]
        self.action_space_shape = self.env.action_space(key).n
        self.model_names = ["RNN0", "RNN0", "RNN1"]
        self.agents = {self.env.agents[i]: self.model_names[i] for i in range(len(self.env.agents))}
        observations = {agent: [] for agent in self.env.agents}
        seq_length = 5
        for i in range(seq_length):
            actions = {agent: self.env.action_space(agent).sample() for agent in self.env.agents}
            obs, rewards, terminations, truncations, infos = self.env.step(actions)
            for agent in self.env.agents:
                observations[agent].append(obs[agent])
        self.observations = {key: np.array(value) for key, value in observations.items()}

        self.avail_actions = self.env.action_space
        self.env.close()

        # Mock model configuration
        self.model_layers = {
            "input_shape": self.obs_shape,
            "rnn_hidden_dim": 128,
            "output_shape": self.action_space_shape
        }

        self.model_configs = {
            "RNN0": ModelConfig(model_type="RNN",layers=self.model_layers),
            "RNN1": ModelConfig(model_type="RNN",layers=self.model_layers)
        }
        
        # Initialize QMIXAgents
        self.agent_group = QMIXAgentGroup(agents=self.agents, model_configs=self.model_configs, device='cpu')
        
        # Mock observations and avail_actions
        #self.observations = [torch.tensor([0.1, 0.2, 0.3]), torch.tensor([0.4, 0.5, 0.6])]
        #self.avail_actions = [torch.tensor([1, 0, 1]), torch.tensor([1, 1, 0])]

        
    def test_get_q_values(self):
        # Test get_q_values method in evaluation mode
        q_values = self.agent_group.get_q_values(observations=self.observations, eval_mode=True)
        self.assertEqual(q_values.shape, (len(self.env.agents), self.action_space_shape))
        
        # Test get_q_values method in training mode
        q_values = self.agent_group.get_q_values(observations=self.observations, eval_mode=False)
        self.assertEqual(q_values.shape, (len(self.env.agents), self.action_space_shape))

    def test_act(self):
        # Test act method with epsilon = 0 (greedy policy)
        actions = self.agent_group.act(self.observations, self.env.action_spaces, epsilon=0)
        self.assertEqual(len(actions), len(self.env.agents))

        # Test act method with epsilon = 1 (random policy)
        actions = self.agent_group.act(self.observations, self.env.action_spaces, epsilon=1)
        self.assertEqual(len(actions), len(self.env.agents))

        # Test act method with epsilon = 0.5
        actions = self.agent_group.act(self.observations, self.env.action_spaces, epsilon=0.5)
        self.assertEqual(len(actions), len(self.env.agents))

if __name__ == '__main__':
    unittest.main()
