import unittest
import torch
import yaml
import numpy as np
import tempfile

from marlite.algorithm.agents import AgentGroupConfig
from marlite.environment import EnvConfig


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
        self.seq_length = 5
        for i in range(self.seq_length):
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
        traj_padding_mask = torch.zeros((bs, self.seq_length))
        alive_mask = torch.ones((bs, len(self.env.agents)))

        # Test get_q_values method in evaluation mode
        ret = self.agent_group.forward(observations=obs, states=states, traj_padding_mask=traj_padding_mask, alive_mask=alive_mask)
        q_values = ret['q_val']
        q_values = q_values.detach().cpu().numpy().squeeze()
        self.assertEqual(q_values.shape, (bs, len(self.env.agents), self.action_space_shape))
        edge_indices = ret['edge_indices']
        self.assertEqual(len(edge_indices), bs)
        self.assertEqual(edge_indices[0].shape[0], 2)

        # Test get_q_values method in training mode
        self.agent_group.train()
        ret = self.agent_group.forward(observations=obs, states=states, traj_padding_mask=traj_padding_mask, alive_mask=alive_mask)
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
        edge_indices = ret['edge_indices']
        self.assertEqual(edge_indices.shape[0], 2)

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


class TestProbObsGNNAgentGroup(unittest.TestCase):

    def setUp(self):
        # Agent group configuration
        config = """
agent_group_config:
  type: "ProbObsGNNComm"
  agent_list:
    predator_0: model1
    predator_1: model1
    predator_2: model1
    predator_3: model1
    predator_4: model1
  model_configs:
    model1:
      feature_extractor:
        model_type: "Custom"
        layers:
        - type: Linear
          in_features: 8
          out_features: 8
      encoder:
        model_type: "CustomConv1D"
        layers:
        - type: Conv1d
          in_channels: 8
          out_channels: 8
          kernel_size: 5
          stride: 1
          padding: 0
        - type: Flatten
        - type: LeakyReLU
      decoder:
        model_type: "Custom"
        layers:
        - type: LeakyReLU
        - type: Linear
          in_features: 16
          out_features: 5
  graph_builder_config:
    type: "FullConn"
    valid_node_list: [0, 1, 2, 3, 4]
  graph_model_config:
    model_type: "GAT"
    input_dim: 8
    hidden_dim: 8
    output_dim: 16
  optimizer:
    type: "Adam"
    lr: 0.0002
    weight_decay: 0.00005
  lr_scheduler:
    type: "ReduceLROnPlateau"
    mode: 'max'
    patience: 3
        """
        config = yaml.safe_load(config)
        self.agent_group_config = AgentGroupConfig(**config['agent_group_config'])

        self.obs_shape = 8
        self.action_space_shape = 5

        # Initialize QMIXAgents
        self.agent_group = self.agent_group_config.get_agent_group()

        # Build observations without env
        self.agent_names = ['predator_0', 'predator_1', 'predator_2', 'predator_3', 'predator_4']
        self.seq_length = 5

        # Create random observations for each agent
        observations = {}
        for agent in self.agent_names:
            # Shape: (seq_length, obs_shape)
            observations[agent] = np.random.randn(self.seq_length, self.obs_shape)
        self.observations = observations

    def test_foward(self):
        bs = 5
        obs = [self.observations[ag] for ag in self.agent_group.agent_model_dict.keys()]
        obs = np.stack(obs)
        obs = np.stack([obs for _ in range(bs)])
        obs = torch.Tensor(obs)
        states = np.random.randn(bs, self.obs_shape)  # Random state for testing
        traj_padding_mask = torch.zeros((bs, self.seq_length))
        alive_mask = torch.ones((bs, len(self.agent_names)))

        # Test get_q_values method in evaluation mode
        ret = self.agent_group.forward(observations=obs, states=states, traj_padding_mask=traj_padding_mask, alive_mask=alive_mask)
        q_values = ret['q_val']
        q_values = q_values.detach().cpu().numpy().squeeze()
        self.assertEqual(q_values.shape, (bs, len(self.agent_names), self.action_space_shape))
        edge_indices = ret['edge_indices']
        self.assertEqual(len(edge_indices), bs)
        self.assertEqual(edge_indices[0].shape[0], 2)

        # Test get_q_values method in training mode
        self.agent_group.train()
        ret = self.agent_group.forward(observations=obs, states=states, traj_padding_mask=traj_padding_mask, alive_mask=alive_mask)
        q_values = ret['q_val']
        q_values = q_values.detach().cpu().numpy().squeeze()
        self.assertEqual(q_values.shape, (bs, len(self.agent_names), self.action_space_shape))

    def test_act(self):
        # Test act method with epsilon = 0 (greedy policy)
        traj_padding_mask = np.zeros(self.seq_length)
        state = np.random.randn(self.obs_shape)  # Random state for testing
        avail_actions = {agent: np.array([True for _ in range(self.action_space_shape)]) for agent in self.agent_names}
        ret = self.agent_group.act(self.observations, state, avail_actions, traj_padding_mask, self.agent_names, epsilon=0)
        actions = ret['actions']
        self.assertEqual(len(actions), len(self.agent_names))
        edge_indices = ret['edge_indices']
        self.assertEqual(edge_indices.shape[0], 2)

        # Test act method with epsilon = 1 (random policy)
        ret = self.agent_group.act(self.observations, state, avail_actions, traj_padding_mask, self.agent_names, epsilon=1)
        actions = ret['actions']
        self.assertEqual(len(actions), len(self.agent_names))

        # Test act method with epsilon = 0.5
        ret = self.agent_group.act(self.observations, state, avail_actions, traj_padding_mask, self.agent_names, epsilon=0.5)
        actions = ret['actions']
        self.assertEqual(len(actions), len(self.agent_names))

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


class TestDualPathObsGNNAgentGroup(unittest.TestCase):

    def setUp(self):
        # Agent group configuration
        config = """
agent_group_config:
  type: "DualPathObsGNNComm"
  agent_list:
    predator_0: model1
    predator_1: model1
    predator_2: model1
    predator_3: model1
    predator_4: model1
  model_configs:
    model1:
      feature_extractor:
        model_type: "Custom"
        layers:
        - type: Linear
          in_features: 8
          out_features: 8
      msg_feature_extractor:
        model_type: "Custom"
        layers:
        - type: Linear
          in_features: 8
          out_features: 8
      encoder:
        model_type: "CustomConv1D"
        layers:
        - type: Conv1d
          in_channels: 8
          out_channels: 8
          kernel_size: 5
          stride: 1
          padding: 0
        - type: Flatten
        - type: LeakyReLU
      decoder:
        model_type: "Custom"
        layers:
        - type: LeakyReLU
        - type: Linear
          in_features: 16
          out_features: 5
  graph_builder_config:
    type: "FullConn"
    valid_node_list: [0, 1, 2, 3, 4]
  graph_model_config:
    model_type: "GAT"
    input_dim: 8
    hidden_dim: 8
    output_dim: 8
  optimizer:
    type: "Adam"
    lr: 0.0002
    weight_decay: 0.00005
  lr_scheduler:
    type: "ReduceLROnPlateau"
    mode: 'max'
    patience: 3
        """
        config = yaml.safe_load(config)
        self.agent_group_config = AgentGroupConfig(**config['agent_group_config'])

        self.obs_shape = 8
        self.action_space_shape = 5

        # Initialize QMIXAgents
        self.agent_group = self.agent_group_config.get_agent_group()

        # Build observations without env
        self.agent_names = ['predator_0', 'predator_1', 'predator_2', 'predator_3', 'predator_4']
        self.seq_length = 5

        # Create random observations for each agent
        observations = {}
        for agent in self.agent_names:
            # Shape: (seq_length, obs_shape)
            observations[agent] = np.random.randn(self.seq_length, self.obs_shape)
        self.observations = observations

    def test_foward(self):
        bs = 5
        obs = [self.observations[ag] for ag in self.agent_group.agent_model_dict.keys()]
        obs = np.stack(obs)
        obs = np.stack([obs for _ in range(bs)])
        obs = torch.Tensor(obs)
        states = np.random.randn(bs, self.obs_shape)  # Random state for testing
        traj_padding_mask = torch.zeros((bs, self.seq_length))
        alive_mask = torch.ones((bs, len(self.agent_names)))

        # Test get_q_values method in evaluation mode
        ret = self.agent_group.forward(observations=obs, states=states, traj_padding_mask=traj_padding_mask, alive_mask=alive_mask)
        q_values = ret['q_val']
        q_values = q_values.detach().cpu().numpy().squeeze()
        self.assertEqual(q_values.shape, (bs, len(self.agent_names), self.action_space_shape))
        edge_indices = ret['edge_indices']
        self.assertEqual(len(edge_indices), bs)
        self.assertEqual(edge_indices[0].shape[0], 2)

        # Test get_q_values method in training mode
        self.agent_group.train()
        ret = self.agent_group.forward(observations=obs, states=states, traj_padding_mask=traj_padding_mask, alive_mask=alive_mask)
        q_values = ret['q_val']
        q_values = q_values.detach().cpu().numpy().squeeze()
        self.assertEqual(q_values.shape, (bs, len(self.agent_names), self.action_space_shape))

    def test_act(self):
        # Test act method with epsilon = 0 (greedy policy)
        traj_padding_mask = np.zeros(self.seq_length)
        state = np.random.randn(self.obs_shape)  # Random state for testing
        avail_actions = {agent: np.array([True for _ in range(self.action_space_shape)]) for agent in self.agent_names}
        ret = self.agent_group.act(self.observations, state, avail_actions, traj_padding_mask, self.agent_names, epsilon=0)
        actions = ret['actions']
        self.assertEqual(len(actions), len(self.agent_names))
        edge_indices = ret['edge_indices']
        self.assertEqual(edge_indices.shape[0], 2)

        # Test act method with epsilon = 1 (random policy)
        ret = self.agent_group.act(self.observations, state, avail_actions, traj_padding_mask, self.agent_names, epsilon=1)
        actions = ret['actions']
        self.assertEqual(len(actions), len(self.agent_names))

        # Test act method with epsilon = 0.5
        ret = self.agent_group.act(self.observations, state, avail_actions, traj_padding_mask, self.agent_names, epsilon=0.5)
        actions = ret['actions']
        self.assertEqual(len(actions), len(self.agent_names))

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


class TestDualPathProbObsGNNAgentGroup(unittest.TestCase):

    def setUp(self):
        # Agent group configuration
        config = """
agent_group_config:
  type: "DualPathProbObsGNNComm"
  agent_list:
    predator_0: model1
    predator_1: model1
    predator_2: model1
    predator_3: model1
    predator_4: model1
  model_configs:
    model1:
      feature_extractor:
        model_type: "Custom"
        layers:
        - type: Linear
          in_features: 8
          out_features: 8
      msg_feature_extractor:
        model_type: "Custom"
        layers:
        - type: Linear
          in_features: 8
          out_features: 8
      encoder:
        model_type: "CustomConv1D"
        layers:
        - type: Conv1d
          in_channels: 8
          out_channels: 8
          kernel_size: 5
          stride: 1
          padding: 0
        - type: Flatten
        - type: LeakyReLU
      decoder:
        model_type: "Custom"
        layers:
        - type: LeakyReLU
        - type: Linear
          in_features: 16
          out_features: 5
  graph_builder_config:
    type: "FullConn"
    valid_node_list: [0, 1, 2, 3, 4]
  graph_model_config:
    model_type: "GAT"
    input_dim: 8
    hidden_dim: 8
    output_dim: 16
  optimizer:
    type: "Adam"
    lr: 0.0002
    weight_decay: 0.00005
  lr_scheduler:
    type: "ReduceLROnPlateau"
    mode: 'max'
    patience: 3
        """
        config = yaml.safe_load(config)
        self.agent_group_config = AgentGroupConfig(**config['agent_group_config'])

        self.obs_shape = 8
        self.action_space_shape = 5

        # Initialize QMIXAgents
        self.agent_group = self.agent_group_config.get_agent_group()

        # Build observations without env
        self.agent_names = ['predator_0', 'predator_1', 'predator_2', 'predator_3', 'predator_4']
        self.seq_length = 5

        # Create random observations for each agent
        observations = {}
        for agent in self.agent_names:
            # Shape: (seq_length, obs_shape)
            observations[agent] = np.random.randn(self.seq_length, self.obs_shape)
        self.observations = observations

    def test_foward(self):
        bs = 5
        obs = [self.observations[ag] for ag in self.agent_group.agent_model_dict.keys()]
        obs = np.stack(obs)
        obs = np.stack([obs for _ in range(bs)])
        obs = torch.Tensor(obs)
        states = np.random.randn(bs, self.obs_shape)  # Random state for testing
        traj_padding_mask = torch.zeros((bs, self.seq_length))
        alive_mask = torch.ones((bs, len(self.agent_names)))

        # Test get_q_values method in evaluation mode
        ret = self.agent_group.forward(observations=obs, states=states, traj_padding_mask=traj_padding_mask, alive_mask=alive_mask)
        q_values = ret['q_val']
        q_values = q_values.detach().cpu().numpy().squeeze()
        self.assertEqual(q_values.shape, (bs, len(self.agent_names), self.action_space_shape))
        edge_indices = ret['edge_indices']
        self.assertEqual(len(edge_indices), bs)
        self.assertEqual(edge_indices[0].shape[0], 2)

        # Test get_q_values method in training mode
        self.agent_group.train()
        ret = self.agent_group.forward(observations=obs, states=states, traj_padding_mask=traj_padding_mask, alive_mask=alive_mask)
        q_values = ret['q_val']
        q_values = q_values.detach().cpu().numpy().squeeze()
        self.assertEqual(q_values.shape, (bs, len(self.agent_names), self.action_space_shape))

    def test_act(self):
        # Test act method with epsilon = 0 (greedy policy)
        traj_padding_mask = np.zeros(self.seq_length)
        state = np.random.randn(self.obs_shape)  # Random state for testing
        avail_actions = {agent: np.array([True for _ in range(self.action_space_shape)]) for agent in self.agent_names}
        ret = self.agent_group.act(self.observations, state, avail_actions, traj_padding_mask, self.agent_names, epsilon=0)
        actions = ret['actions']
        self.assertEqual(len(actions), len(self.agent_names))
        edge_indices = ret['edge_indices']
        self.assertEqual(edge_indices.shape[0], 2)

        # Test act method with epsilon = 1 (random policy)
        ret = self.agent_group.act(self.observations, state, avail_actions, traj_padding_mask, self.agent_names, epsilon=1)
        actions = ret['actions']
        self.assertEqual(len(actions), len(self.agent_names))

        # Test act method with epsilon = 0.5
        ret = self.agent_group.act(self.observations, state, avail_actions, traj_padding_mask, self.agent_names, epsilon=0.5)
        actions = ret['actions']
        self.assertEqual(len(actions), len(self.agent_names))

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
