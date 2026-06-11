import unittest
import torch
import yaml
import numpy as np

from marlite.algorithm.agents import AgentGroupConfig
from marlite.environment import EnvConfig


class TestGNNAgentGroup(unittest.TestCase):
    def setUp(self):
        # Agent group configuration
        config = yaml.safe_load("""
agent_group:
  type: "GNN"
  agent_list:
    predator_0: model1
    predator_1: model1
    predator_2: model1
    predator_3: model1
    predator_4: model1
    predator_5: model1
    predator_6: model1
    predator_7: model1
    predator_8: model1
    predator_9: model1
    predator_10: model1
    predator_11: model1
    predator_12: model1
    predator_13: model1
    predator_14: model1
    predator_15: model1
    predator_16: model1
    predator_17: model1
    predator_18: model1
    predator_19: model1
    predator_20: model1
    predator_21: model1
    predator_22: model1
    predator_23: model1
    predator_24: model1
  models:
    model1:
      feature_extractor:
        model_type: "Custom"
        layers:
        - type: Conv2d
          in_channels: 29
          out_channels: 8
          kernel_size: 1
          stride: 1
          padding: 0
        - type: LeakyReLU
        - type: Conv2d
          in_channels: 8
          out_channels: 1
          kernel_size: 3
          stride: 1
          padding: 0
        - type: BatchNorm2d
          num_features: 1
        - type: LeakyReLU
        - type: Flatten
      encoder:
        model_type: "CustomConv1D"
        layers:
        - type: Conv1d
          in_channels: 64
          out_channels: 64
          kernel_size: 3
          stride: 2
          padding: 0
        - type: Flatten
        - type: LeakyReLU
      decoder:
        model_type: "Custom"
        layers:
        - type: LeakyReLU
        - type: Linear
          in_features: 64
          out_features: 13
  graph_builder:
    type: "PartialMAgent"
    binary_agent_id_dim: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    agent_presence_dim: [1, 3]
    comm_distance: 20
    distance_metric: "cityblock"
    n_workers: 20
    n_subgraphs: 5
    valid_node_list: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    target_node_list: [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]
    update_interval: 5
  graph_model:
    model_type: "GAT"
    input_dim: 128
    hidden_dim: 128
    output_dim: 64
  optimizer:
    type: "Adam"
    lr: 0.0002
    weight_decay: 0.00005
  lr_scheduler:
    type: "ReduceLROnPlateau"
    mode: "max"
    patience: 3
env_config:
  module_name: "magent2.environments"
  env_name: "adversarial_pursuit_v4"
  env_params:
    tag_penalty: 0.0
    extra_features: true
  wrapper:
    type: adversarial_pursuit_predator
    opponent_agent_group:
      type: "MAgentPrey"
      agent_list:
        prey_0: random1
        prey_1: random1
        prey_2: random1
        prey_3: random1
        prey_4: random1
        prey_5: random1
        prey_6: random1
        prey_7: random1
        prey_8: random1
        prey_9: random1
        prey_10: random1
        prey_11: random1
        prey_12: random1
        prey_13: random1
        prey_14: random1
        prey_15: random1
        prey_16: random1
        prey_17: random1
        prey_18: random1
        prey_19: random1
        prey_20: random1
        prey_21: random1
        prey_22: random1
        prey_23: random1
        prey_24: random1
        prey_25: random1
        prey_26: random1
        prey_27: random1
        prey_28: random1
        prey_29: random1
        prey_30: random1
        prey_31: random1
        prey_32: random1
        prey_33: random1
        prey_34: random1
        prey_35: random1
        prey_36: random1
        prey_37: random1
        prey_38: random1
        prey_39: random1
        prey_40: random1
        prey_41: random1
        prey_42: random1
        prey_43: random1
        prey_44: random1
        prey_45: random1
        prey_46: random1
        prey_47: random1
        prey_48: random1
        prey_49: random1
    opp_obs_queue_len: 1
    channel_first: true
""")
        self.agent_group_config = AgentGroupConfig(**config["agent_group"])

        # Environment setup and model configuration
        self.env_config = EnvConfig(**config["env_config"])
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
            actions = {
                agent: self.env.action_space(agent).sample()
                for agent in self.env.agents
            }
            obs, rewards, terminations, truncations, infos = self.env.step(actions)
            for agent in self.env.agents:
                observations[agent].append(obs[agent])
        self.observations = {
            key: np.array(value) for key, value in observations.items()
        }

    def test_foward(self):
        bs = 5
        obs = [self.observations[ag] for ag in self.agent_group.agent_model_dict.keys()]
        obs = np.stack(obs)
        obs = np.stack([obs for _ in range(bs)])
        obs = torch.Tensor(obs)
        states = np.stack([self.env.state() for _ in range(bs)])
        states = torch.from_numpy(states).float()
        traj_padding_mask = torch.zeros((bs, self.seq_length))
        alive_mask = torch.ones((bs, len(self.env.agents)))

        # Test get_q_values method in evaluation mode
        ret = self.agent_group.forward(
            observations=obs,
            states=states,
            traj_padding_mask=traj_padding_mask,
            alive_mask=alive_mask,
        )
        q_values = ret["q_val"]
        q_values = q_values.detach().cpu().numpy().squeeze()
        self.assertEqual(
            q_values.shape, (bs, len(self.env.agents), self.action_space_shape)
        )
        edge_indices = ret["edge_indices"]
        self.assertEqual(len(edge_indices), bs)
        self.assertEqual(edge_indices[0].shape[0], 2)

        # Test get_q_values method in training mode
        self.agent_group.train()
        ret = self.agent_group.forward(
            observations=obs,
            states=states,
            traj_padding_mask=traj_padding_mask,
            alive_mask=alive_mask,
        )
        q_values = ret["q_val"]
        q_values = q_values.detach().cpu().numpy().squeeze()
        self.assertEqual(
            q_values.shape, (bs, len(self.env.agents), self.action_space_shape)
        )

    def test_act(self):
        # Test act method with epsilon = 0 (greedy policy)
        traj_padding_mask = np.zeros(self.seq_length)
        state = self.env.state()
        ret = self.agent_group.act(
            self.observations,
            state,
            self.env.action_spaces,
            traj_padding_mask,
            self.env.agents,
            epsilon=0,
        )
        actions = ret["actions"]
        self.assertEqual(len(actions), len(self.env.agents))
        edge_indices = ret["edge_indices"]
        self.assertEqual(edge_indices.shape[0], 2)

        # Test act method with epsilon = 1 (random policy)
        ret = self.agent_group.act(
            self.observations,
            state,
            self.env.action_spaces,
            traj_padding_mask,
            self.env.agents,
            epsilon=1,
        )
        actions = ret["actions"]
        self.assertEqual(len(actions), len(self.env.agents))

        # Test act method with epsilon = 0.5
        ret = self.agent_group.act(
            self.observations,
            state,
            self.env.action_spaces,
            traj_padding_mask,
            self.env.agents,
            epsilon=0.5,
        )
        actions = ret["actions"]
        self.assertEqual(len(actions), len(self.env.agents))

    def test_eval(self):
        self.agent_group.eval()
        # Check if the agent group is in evaluation mode
        for (_, fe), (_, encoder), (_, decoder) in zip(
            self.agent_group.feature_extractors.items(),
            self.agent_group.encoders.items(),
            self.agent_group.decoders.items(),
        ):
            self.assertFalse(fe.training)
            self.assertFalse(encoder.training)
            self.assertFalse(decoder.training)

    def test_train(self):
        self.agent_group.train()
        # Check if the agent group is in evaluation mode
        for (_, fe), (_, encoder), (_, decoder) in zip(
            self.agent_group.feature_extractors.items(),
            self.agent_group.encoders.items(),
            self.agent_group.decoders.items(),
        ):
            self.assertTrue(fe.training)
            self.assertTrue(encoder.training)
            self.assertTrue(decoder.training)


class TestProbObsGNNAgentGroup(unittest.TestCase):
    def setUp(self):
        # Agent group configuration
        config = """
agent_group:
  type: "ProbObsGNNComm"
  agent_list:
    predator_0: model1
    predator_1: model1
    predator_2: model1
    predator_3: model1
    predator_4: model1
  models:
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
  graph_builder:
    type: "FullConn"
    valid_node_list: [0, 1, 2, 3, 4]
  graph_model:
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
    mode: "max"
    patience: 3
        """
        config = yaml.safe_load(config)
        self.agent_group_config = AgentGroupConfig(**config["agent_group"])

        self.obs_shape = 8
        self.action_space_shape = 5

        # Initialize QMIXAgents
        self.agent_group = self.agent_group_config.get_agent_group()

        # Build observations without env
        self.agent_names = [
            "predator_0",
            "predator_1",
            "predator_2",
            "predator_3",
            "predator_4",
        ]
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
        states = torch.from_numpy(states).float()
        traj_padding_mask = torch.zeros((bs, self.seq_length))
        alive_mask = torch.ones((bs, len(self.agent_names)))

        # Test get_q_values method in evaluation mode
        ret = self.agent_group.forward(
            observations=obs,
            states=states,
            traj_padding_mask=traj_padding_mask,
            alive_mask=alive_mask,
        )
        q_values = ret["q_val"]
        q_values = q_values.detach().cpu().numpy().squeeze()
        self.assertEqual(
            q_values.shape, (bs, len(self.agent_names), self.action_space_shape)
        )
        edge_indices = ret["edge_indices"]
        self.assertEqual(len(edge_indices), bs)
        self.assertEqual(edge_indices[0].shape[0], 2)

        # Test get_q_values method in training mode
        self.agent_group.train()
        ret = self.agent_group.forward(
            observations=obs,
            states=states,
            traj_padding_mask=traj_padding_mask,
            alive_mask=alive_mask,
        )
        q_values = ret["q_val"]
        q_values = q_values.detach().cpu().numpy().squeeze()
        self.assertEqual(
            q_values.shape, (bs, len(self.agent_names), self.action_space_shape)
        )

    def test_act(self):
        # Test act method with epsilon = 0 (greedy policy)
        traj_padding_mask = np.zeros(self.seq_length)
        state = np.random.randn(self.obs_shape)  # Random state for testing
        avail_actions = {
            agent: np.array([True for _ in range(self.action_space_shape)])
            for agent in self.agent_names
        }
        ret = self.agent_group.act(
            self.observations,
            state,
            avail_actions,
            traj_padding_mask,
            self.agent_names,
            epsilon=0,
        )
        actions = ret["actions"]
        self.assertEqual(len(actions), len(self.agent_names))
        edge_indices = ret["edge_indices"]
        self.assertEqual(edge_indices.shape[0], 2)

        # Test act method with epsilon = 1 (random policy)
        ret = self.agent_group.act(
            self.observations,
            state,
            avail_actions,
            traj_padding_mask,
            self.agent_names,
            epsilon=1,
        )
        actions = ret["actions"]
        self.assertEqual(len(actions), len(self.agent_names))

        # Test act method with epsilon = 0.5
        ret = self.agent_group.act(
            self.observations,
            state,
            avail_actions,
            traj_padding_mask,
            self.agent_names,
            epsilon=0.5,
        )
        actions = ret["actions"]
        self.assertEqual(len(actions), len(self.agent_names))

    def test_eval(self):
        self.agent_group.eval()
        # Check if the agent group is in evaluation mode
        for (_, fe), (_, encoder), (_, decoder) in zip(
            self.agent_group.feature_extractors.items(),
            self.agent_group.encoders.items(),
            self.agent_group.decoders.items(),
        ):
            self.assertFalse(fe.training)
            self.assertFalse(encoder.training)
            self.assertFalse(decoder.training)

    def test_train(self):
        self.agent_group.train()
        # Check if the agent group is in evaluation mode
        for (_, fe), (_, encoder), (_, decoder) in zip(
            self.agent_group.feature_extractors.items(),
            self.agent_group.encoders.items(),
            self.agent_group.decoders.items(),
        ):
            self.assertTrue(fe.training)
            self.assertTrue(encoder.training)
            self.assertTrue(decoder.training)


class TestDualPathObsGNNAgentGroup(unittest.TestCase):
    def setUp(self):
        # Agent group configuration
        config = """
agent_group:
  type: "DualPathObsGNNComm"
  agent_list:
    predator_0: model1
    predator_1: model1
    predator_2: model1
    predator_3: model1
    predator_4: model1
  models:
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
  graph_builder:
    type: "FullConn"
    valid_node_list: [0, 1, 2, 3, 4]
  graph_model:
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
    mode: "max"
    patience: 3
        """
        config = yaml.safe_load(config)
        self.agent_group_config = AgentGroupConfig(**config["agent_group"])

        self.obs_shape = 8
        self.action_space_shape = 5

        # Initialize QMIXAgents
        self.agent_group = self.agent_group_config.get_agent_group()

        # Build observations without env
        self.agent_names = [
            "predator_0",
            "predator_1",
            "predator_2",
            "predator_3",
            "predator_4",
        ]
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
        states = torch.from_numpy(states).float()
        traj_padding_mask = torch.zeros((bs, self.seq_length))
        alive_mask = torch.ones((bs, len(self.agent_names)))

        # Test get_q_values method in evaluation mode
        ret = self.agent_group.forward(
            observations=obs,
            states=states,
            traj_padding_mask=traj_padding_mask,
            alive_mask=alive_mask,
        )
        q_values = ret["q_val"]
        q_values = q_values.detach().cpu().numpy().squeeze()
        self.assertEqual(
            q_values.shape, (bs, len(self.agent_names), self.action_space_shape)
        )
        edge_indices = ret["edge_indices"]
        self.assertEqual(len(edge_indices), bs)
        self.assertEqual(edge_indices[0].shape[0], 2)

        # Test get_q_values method in training mode
        self.agent_group.train()
        ret = self.agent_group.forward(
            observations=obs,
            states=states,
            traj_padding_mask=traj_padding_mask,
            alive_mask=alive_mask,
        )
        q_values = ret["q_val"]
        q_values = q_values.detach().cpu().numpy().squeeze()
        self.assertEqual(
            q_values.shape, (bs, len(self.agent_names), self.action_space_shape)
        )

    def test_act(self):
        # Test act method with epsilon = 0 (greedy policy)
        traj_padding_mask = np.zeros(self.seq_length)
        state = np.random.randn(self.obs_shape)  # Random state for testing
        avail_actions = {
            agent: np.array([True for _ in range(self.action_space_shape)])
            for agent in self.agent_names
        }
        ret = self.agent_group.act(
            self.observations,
            state,
            avail_actions,
            traj_padding_mask,
            self.agent_names,
            epsilon=0,
        )
        actions = ret["actions"]
        self.assertEqual(len(actions), len(self.agent_names))
        edge_indices = ret["edge_indices"]
        self.assertEqual(edge_indices.shape[0], 2)

        # Test act method with epsilon = 1 (random policy)
        ret = self.agent_group.act(
            self.observations,
            state,
            avail_actions,
            traj_padding_mask,
            self.agent_names,
            epsilon=1,
        )
        actions = ret["actions"]
        self.assertEqual(len(actions), len(self.agent_names))

        # Test act method with epsilon = 0.5
        ret = self.agent_group.act(
            self.observations,
            state,
            avail_actions,
            traj_padding_mask,
            self.agent_names,
            epsilon=0.5,
        )
        actions = ret["actions"]
        self.assertEqual(len(actions), len(self.agent_names))

    def test_eval(self):
        self.agent_group.eval()
        # Check if the agent group is in evaluation mode
        for (_, fe), (_, encoder), (_, decoder) in zip(
            self.agent_group.feature_extractors.items(),
            self.agent_group.encoders.items(),
            self.agent_group.decoders.items(),
        ):
            self.assertFalse(fe.training)
            self.assertFalse(encoder.training)
            self.assertFalse(decoder.training)

    def test_train(self):
        self.agent_group.train()
        # Check if the agent group is in evaluation mode
        for (_, fe), (_, encoder), (_, decoder) in zip(
            self.agent_group.feature_extractors.items(),
            self.agent_group.encoders.items(),
            self.agent_group.decoders.items(),
        ):
            self.assertTrue(fe.training)
            self.assertTrue(encoder.training)
            self.assertTrue(decoder.training)


class TestDualPathProbObsGNNAgentGroup(unittest.TestCase):
    def setUp(self):
        # Agent group configuration
        config = """
agent_group:
  type: "DualPathProbObsGNNComm"
  agent_list:
    predator_0: model1
    predator_1: model1
    predator_2: model1
    predator_3: model1
    predator_4: model1
  models:
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
  graph_builder:
    type: "FullConn"
    valid_node_list: [0, 1, 2, 3, 4]
  graph_model:
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
    mode: "max"
    patience: 3
        """
        config = yaml.safe_load(config)
        self.agent_group_config = AgentGroupConfig(**config["agent_group"])

        self.obs_shape = 8
        self.action_space_shape = 5

        # Initialize QMIXAgents
        self.agent_group = self.agent_group_config.get_agent_group()

        # Build observations without env
        self.agent_names = [
            "predator_0",
            "predator_1",
            "predator_2",
            "predator_3",
            "predator_4",
        ]
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
        states = torch.from_numpy(states).float()
        traj_padding_mask = torch.zeros((bs, self.seq_length))
        alive_mask = torch.ones((bs, len(self.agent_names)))

        # Test get_q_values method in evaluation mode
        ret = self.agent_group.forward(
            observations=obs,
            states=states,
            traj_padding_mask=traj_padding_mask,
            alive_mask=alive_mask,
        )
        q_values = ret["q_val"]
        q_values = q_values.detach().cpu().numpy().squeeze()
        self.assertEqual(
            q_values.shape, (bs, len(self.agent_names), self.action_space_shape)
        )
        edge_indices = ret["edge_indices"]
        self.assertEqual(len(edge_indices), bs)
        self.assertEqual(edge_indices[0].shape[0], 2)

        # Test get_q_values method in training mode
        self.agent_group.train()
        ret = self.agent_group.forward(
            observations=obs,
            states=states,
            traj_padding_mask=traj_padding_mask,
            alive_mask=alive_mask,
        )
        q_values = ret["q_val"]
        q_values = q_values.detach().cpu().numpy().squeeze()
        self.assertEqual(
            q_values.shape, (bs, len(self.agent_names), self.action_space_shape)
        )

    def test_act(self):
        # Test act method with epsilon = 0 (greedy policy)
        traj_padding_mask = np.zeros(self.seq_length)
        state = np.random.randn(self.obs_shape)  # Random state for testing
        avail_actions = {
            agent: np.array([True for _ in range(self.action_space_shape)])
            for agent in self.agent_names
        }
        ret = self.agent_group.act(
            self.observations,
            state,
            avail_actions,
            traj_padding_mask,
            self.agent_names,
            epsilon=0,
        )
        actions = ret["actions"]
        self.assertEqual(len(actions), len(self.agent_names))
        edge_indices = ret["edge_indices"]
        self.assertEqual(edge_indices.shape[0], 2)

        # Test act method with epsilon = 1 (random policy)
        ret = self.agent_group.act(
            self.observations,
            state,
            avail_actions,
            traj_padding_mask,
            self.agent_names,
            epsilon=1,
        )
        actions = ret["actions"]
        self.assertEqual(len(actions), len(self.agent_names))

        # Test act method with epsilon = 0.5
        ret = self.agent_group.act(
            self.observations,
            state,
            avail_actions,
            traj_padding_mask,
            self.agent_names,
            epsilon=0.5,
        )
        actions = ret["actions"]
        self.assertEqual(len(actions), len(self.agent_names))

    def test_eval(self):
        self.agent_group.eval()
        # Check if the agent group is in evaluation mode
        for (_, fe), (_, encoder), (_, decoder) in zip(
            self.agent_group.feature_extractors.items(),
            self.agent_group.encoders.items(),
            self.agent_group.decoders.items(),
        ):
            self.assertFalse(fe.training)
            self.assertFalse(encoder.training)
            self.assertFalse(decoder.training)

    def test_train(self):
        self.agent_group.train()
        # Check if the agent group is in evaluation mode
        for (_, fe), (_, encoder), (_, decoder) in zip(
            self.agent_group.feature_extractors.items(),
            self.agent_group.encoders.items(),
            self.agent_group.decoders.items(),
        ):
            self.assertTrue(fe.training)
            self.assertTrue(encoder.training)
            self.assertTrue(decoder.training)


if __name__ == "__main__":
    unittest.main()
