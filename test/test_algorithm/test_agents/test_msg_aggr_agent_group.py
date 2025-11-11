import unittest
import torch
import yaml
import numpy as np
import tempfile

from marlite.algorithm.agents import AgentGroupConfig
from marlite.environment import EnvConfig


class TestSeqMsgAggrAgentGroup(unittest.TestCase):

    def setUp(self):
        # Agent group configuration
        config_path = 'test/config/seq_msg_aggr_smac.yaml'
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

        self.seq_length = config['rollout_config']['traj_len']

        # Initialize QMIXAgents
        self.agent_group = self.agent_group_config.get_agent_group()
        obs, info = self.env.reset()

        self.observations = {agent: np.stack([obs[agent] for _ in range(self.seq_length)], axis=0) for agent in self.env.agents}

    def test_foward(self):
        bs = 5
        n_agents = len(self.agent_group.agent_model_dict)
        obs = [self.observations[ag] for ag in self.agent_group.agent_model_dict.keys()]
        obs = np.stack(obs)
        obs = np.stack([obs for _ in range(bs)])
        obs = torch.Tensor(obs)
        states = np.stack([self.env.state() for _ in range(bs)])
        traj_padding_mask = torch.ones((bs, n_agents, self.seq_length))
        traj_padding_mask[:, 0] = torch.zeros_like(traj_padding_mask[:, 0])
        traj_padding_mask = traj_padding_mask.to(dtype=torch.bool)
        alive_mask = torch.ones((bs, len(self.env.agents)))

        # Test get_q_values method in evaluation mode
        ret = self.agent_group.forward(observations=obs, traj_padding_mask=traj_padding_mask, alive_mask=alive_mask)
        q_values = ret['q_val']
        q_values = q_values.detach().cpu().numpy().squeeze()
        self.assertEqual(q_values.shape, (bs, len(self.env.agents), self.action_space_shape))

        # Test get_q_values method in training mode
        self.agent_group.train()
        ret = self.agent_group.forward(observations=obs, traj_padding_mask=traj_padding_mask, alive_mask=alive_mask)
        q_values = ret['q_val']
        q_values = q_values.detach().cpu().numpy().squeeze()
        self.assertEqual(q_values.shape, (bs, len(self.env.agents), self.action_space_shape))

    def test_act(self):
        # Test act method with epsilon = 0 (greedy policy)
        traj_padding_mask = np.zeros(self.seq_length)
        state = self.env.state()
        ret = self.agent_group.act(self.observations, state, self.env.action_spaces, traj_padding_mask, self.env.agents, epsilon=0)
        actions = ret['actions']
        self.assertEqual(len(actions), len(self.env.agents))

        # Test act method with epsilon = 1 (random policy)
        ret = self.agent_group.act(self.observations, state, self.env.action_spaces, traj_padding_mask, self.env.agents, epsilon=1)
        actions = ret['actions']
        self.assertEqual(len(actions), len(self.env.agents))

        # Test act method with epsilon = 0.5
        ret = self.agent_group.act(self.observations, state, self.env.action_spaces, traj_padding_mask, self.env.agents, epsilon=0.5)
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

    def test_lr_scheduler_step(self):
        self.agent_group.lr_scheduler_step(0)
        self.assertIsNotNone(self.agent_group.lr_scheduler)

if __name__ == '__main__':
    unittest.main()
