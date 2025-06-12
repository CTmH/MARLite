import unittest
import torch
import yaml
import numpy as np
import tempfile
from unittest.mock import MagicMock
from pettingzoo.mpe import simple_spread_v3

from src.algorithm.agents import QMIXAgentGroup
from src.algorithm.agents.agent_group_config import AgentGroupConfig
from src.algorithm.model import ModelConfig
from src.util.optimizer_config import OptimizerConfig


class TestQMIXAgentGroup(unittest.TestCase):

    def setUp(self):
        # Environment setup and model configuration
        self.env = simple_spread_v3.parallel_env(render_mode="human")
        obs, _ = self.env.reset()
        key = self.env.agents[0]
        self.obs_shape = self.env.observation_space(key).shape
        self.obs_shape = self.obs_shape[0]
        self.action_space_shape = self.env.action_space(key).n
        # Agent group configuration
        config_path = 'test/config/qmix_default.yaml'
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.agent_group_config = AgentGroupConfig(**config['agent_group_config'])
        
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
        
    def test_forward(self):
        obs = [self.observations[ag] for ag in self.agent_group.agent_model_dict.keys()]
        obs = np.stack(obs)
        obs = np.expand_dims(obs, axis=0)
        obs = torch.Tensor(obs)
        # Test get_q_values method in evaluation mode
        ret = self.agent_group.forward(observations=obs)
        q_values = ret['q_val']
        q_values = q_values.detach().cpu().numpy().squeeze()
        self.assertEqual(q_values.shape, (len(self.env.agents), self.action_space_shape))
        
        # Test get_q_values method in training mode
        self.agent_group.train()
        ret = self.agent_group.forward(observations=obs)
        q_values = ret['q_val']
        q_values = q_values.detach().cpu().numpy().squeeze()
        self.assertEqual(q_values.shape, (len(self.env.agents), self.action_space_shape))

    def test_act(self):
        # Test act method with epsilon = 0 (greedy policy)
        ret = self.agent_group.act(self.observations, self.env.action_spaces, epsilon=0)
        actions = ret['actions']
        self.assertEqual(len(actions), len(self.env.agents))

        # Test act method with epsilon = 1 (random policy)
        ret = self.agent_group.act(self.observations, self.env.action_spaces, epsilon=1)
        actions = ret['actions']
        self.assertEqual(len(actions), len(self.env.agents))

        # Test act method with epsilon = 0.5
        ret = self.agent_group.act(self.observations, self.env.action_spaces, epsilon=0.5)
        actions = ret['actions']
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

    def test_save_load_params(self):
        # Create a temporary directory to save parameters
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Save the agent group parameters
            self.agent_group.save_params(tmpdirname)
            self.agent_group.load_params(tmpdirname)
        
if __name__ == '__main__':
    unittest.main()
