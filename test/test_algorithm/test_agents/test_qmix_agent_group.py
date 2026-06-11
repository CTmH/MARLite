import unittest
import torch
import yaml
import numpy as np
from mpe2 import simple_spread_v3
from torch.nn.parallel import DistributedDataParallel as DDP
from marlite.algorithm.agents import AgentGroupConfig


class TestQMIXAgentGroup(unittest.TestCase):
    def setUp(self):
        # Environment setup and model configuration
        self.env = simple_spread_v3.parallel_env(render_mode="rgb_array")
        obs, _ = self.env.reset()
        key = self.env.agents[0]
        self.obs_shape = self.env.observation_space(key).shape
        self.obs_shape = self.obs_shape[0]
        self.action_space_shape = self.env.action_space(key).n
        # Agent group configuration
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
        self.agent_group_config = AgentGroupConfig(**config["agent_group"])

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

    def test_forward(self):
        bs = 5
        obs = [self.observations[ag] for ag in self.agent_group.agent_model_dict.keys()]
        obs = np.stack(obs)
        obs = np.stack([obs for _ in range(bs)])
        obs = torch.Tensor(obs)
        states = np.stack([self.env.state() for _ in range(bs)])
        traj_padding_mask = torch.zeros((bs, self.seq_length))
        alive_mask = torch.ones((bs, len(self.env.agents)))

        # Test get_q_values method in evaluation mode
        ret = self.agent_group.forward(
            observations=obs, traj_padding_mask=traj_padding_mask, alive_mask=alive_mask
        )
        q_values = ret["q_val"]
        q_values = q_values.detach().cpu().numpy().squeeze()
        self.assertEqual(
            q_values.shape, (bs, len(self.env.agents), self.action_space_shape)
        )

        # Test get_q_values method in training mode
        self.agent_group.train()
        ret = self.agent_group.forward(
            observations=obs, traj_padding_mask=traj_padding_mask, alive_mask=alive_mask
        )
        q_values = ret["q_val"]
        q_values = q_values.detach().cpu().numpy().squeeze()
        self.assertEqual(
            q_values.shape, (bs, len(self.env.agents), self.action_space_shape)
        )

    def test_dead_agents_q_are_zero(self):
        bs = 3
        obs = [self.observations[ag] for ag in self.agent_group.agent_model_dict.keys()]
        obs = np.stack(obs)
        obs = np.stack([obs for _ in range(bs)])
        obs = torch.Tensor(obs)
        states = np.stack([self.env.state() for _ in range(bs)])
        traj_padding_mask = torch.zeros((bs, self.seq_length))
        alive_mask = torch.ones((bs, len(self.env.agents)))
        alive_mask[:, -1] = 0  # last agent is dead

        ret = self.agent_group.forward(
            observations=obs, traj_padding_mask=traj_padding_mask, alive_mask=alive_mask
        )
        q_values = ret["q_val"]
        dead_q = q_values[:, -1, :]
        self.assertTrue(torch.all(dead_q == 0), f"Dead agent Q-values should be zero, got {dead_q}")

    def test_act(self):
        # Test act method with epsilon = 0 (greedy policy)
        traj_padding_mask = np.ones(self.seq_length)
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
        for (model_name, model), (_, fe) in zip(
            self.agent_group.encoders.items(), self.agent_group.feature_extractors.items()
        ):
            self.assertFalse(model.training)
            self.assertFalse(fe.training)

    def test_train(self):
        self.agent_group.train()
        for (model_name, model), (_, fe) in zip(
            self.agent_group.encoders.items(), self.agent_group.feature_extractors.items()
        ):
            self.assertTrue(model.training)
            self.assertTrue(fe.training)


if __name__ == "__main__":
    unittest.main()
