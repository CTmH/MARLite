import unittest
import torch
import yaml
import numpy as np
import tempfile

from src.algorithm.agents.agent_group_config import AgentGroupConfig
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

    def test_foward(self):
        bs = 5
        obs = [self.observations[ag] for ag in self.agent_group.agent_model_dict.keys()]
        obs = np.stack(obs)
        obs = np.stack([obs for _ in range(bs)])
        obs = torch.Tensor(obs)
        states = np.stack([self.env.state() for _ in range(bs)])

        # Test get_q_values method in evaluation mode
        ret = self.agent_group.forward(observations=obs, states=states)
        q_values = ret['q_val']
        q_values = q_values.detach().cpu().numpy().squeeze()
        self.assertEqual(q_values.shape, (bs, len(self.env.agents), self.action_space_shape))
        edge_indices = ret['edge_indices']
        self.assertEqual(len(edge_indices), bs)
        self.assertEqual(edge_indices[0].shape[0], 2)
        
        # Test get_q_values method in training mode
        self.agent_group.train()
        ret = self.agent_group.forward(observations=obs, states=states)
        q_values = ret['q_val']
        q_values = q_values.detach().cpu().numpy().squeeze()
        self.assertEqual(q_values.shape, (bs, len(self.env.agents), self.action_space_shape))

    def test_act(self):
        # Test act method with epsilon = 0 (greedy policy)
        state = self.env.state()
        ret = self.agent_group.act(self.observations, state, self.env.action_spaces, epsilon=0)
        actions = ret['actions']
        self.assertEqual(len(actions), len(self.env.agents))
        edge_indices = ret['edge_indices']
        self.assertEqual(edge_indices.shape[0], 2)

        # Test act method with epsilon = 1 (random policy)
        ret = self.agent_group.act(self.observations, state, self.env.action_spaces, epsilon=1)
        actions = ret['actions']
        self.assertEqual(len(actions), len(self.env.agents))

        # Test act method with epsilon = 0.5
        ret = self.agent_group.act(self.observations, state, self.env.action_spaces, epsilon=0.5)
        actions = ret['actions']
        self.assertEqual(len(actions), len(self.env.agents))

    def test_eval(self):
        self.agent_group.eval()
        # Check if the agent group is in evaluation mode
        for (_, fe), (_, encoder), (_, decoder)  in zip(
                                                        self.agent_group.feature_extractors.items(),
                                                        self.agent_group.encoders.items(),
                                                        self.agent_group.decoders.items()):
            self.assertFalse(fe.training)
            self.assertFalse(encoder.training)
            self.assertFalse(decoder.training)

    def test_train(self):
        self.agent_group.train()
        # Check if the agent group is in evaluation mode
        for (_, fe), (_, encoder), (_, decoder)  in zip(
                                                        self.agent_group.feature_extractors.items(),
                                                        self.agent_group.encoders.items(),
                                                        self.agent_group.decoders.items()):
            self.assertTrue(fe.training)
            self.assertTrue(encoder.training)
            self.assertTrue(decoder.training)

    def test_save_load_params(self):
        # Create a temporary directory to save parameters
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Save the agent group parameters
            self.agent_group.save_params(tmpdirname)
            self.agent_group.load_params(tmpdirname)

if __name__ == '__main__':
    unittest.main()
