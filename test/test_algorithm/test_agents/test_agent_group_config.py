import unittest
import yaml
from mpe2 import simple_spread_v3

from marlite.algorithm.agents import AgentGroupConfig, AgentGroup
from marlite.algorithm.agents.group_consensus_agent_group import GroupConsensusAgentGroup


class TestAgentGroupConfig(unittest.TestCase):

    def setUp(self):
        self.env = simple_spread_v3.parallel_env(render_mode="human")

    def test_get_agent_group(self):
        config = yaml.safe_load("""
agent_group:
  type: "QMIX"
  agent_list:
    agent_0: model1
    agent_1: model1
    agent_2: model1
  models:
    model1:
      feature_extractor:
        model_type: "Identity"
      encoder:
        model_type: "RNN"
        input_shape: 18
        rnn_hidden_dim: 16
        rnn_layers: 1
        output_shape: 16
      decoder:
        model_type: "Custom"
        layers:
        - type: Linear
          in_features: 16
          out_features: 5
  optimizer:
    type: "Adam"
    lr: 0.0005
    weight_decay: 0.0001
""")
        self.agent_group_config = AgentGroupConfig(**config['agent_group'])
        self.agent_group = self.agent_group_config.get_agent_group()
        self.assertIsInstance(self.agent_group, AgentGroup)

    def test_get_gnn_agent_group(self):
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
""")
        self.agent_group_config = AgentGroupConfig(**config['agent_group'])
        self.agent_group = self.agent_group_config.get_agent_group()
        self.assertIsInstance(self.agent_group, AgentGroup)

    def test_get_group_consensus_agent_group(self):
        config = yaml.safe_load("""
agent_group:
  type: "GroupConsensusQMIX"
  agent_list:
    agent_0: model1
    agent_1: model1
    agent_2: model1
  models:
    model1:
      feature_extractor:
        model_type: "Custom"
        layers:
        - type: Linear
          in_features: 18
          out_features: 32
      group_estimate_feature_extractor:
        model_type: "Custom"
        layers:
        - type: Linear
          in_features: 18
          out_features: 16
      encoder:
        model_type: "Custom"
        layers:
        - type: Linear
          in_features: 40
          out_features: 128
      decoder:
        model_type: "Custom"
        layers:
        - type: Linear
          in_features: 128
          out_features: 5
  group_builder:
    type: "Fixed"
    group_ids: [0, 0, 1]
  deterministic_eval: true
  enable_rl_grad_to_group_estimate: false
  optimizer:
    type: "Adam"
    lr: 0.0005
    weight_decay: 0.0001
""")
        self.agent_group_config = AgentGroupConfig(**config['agent_group'])
        self.agent_group = self.agent_group_config.get_agent_group()
        self.assertIsInstance(self.agent_group, GroupConsensusAgentGroup)


if __name__ == '__main__':
    unittest.main()
