import unittest
import numpy as np
import yaml
import torch
import torch.nn.functional as F

from marlite.algorithm.agents import AgentGroupConfig
from marlite.algorithm.agents.group_consensus_agent_group import GroupConsensusAgentGroup


class FakeAgentGroup:
    device = torch.device("cpu")


def _merge_bayesian_reference(agent_mu, agent_log_var, group_indices, device):
    bs, n_agents, f_z = agent_mu.shape
    G = int(group_indices.max()) + 1
    gids = torch.as_tensor(group_indices, dtype=torch.long, device=device)

    group_mu = agent_mu.new_zeros(bs, G, f_z)
    group_log_var = agent_log_var.new_zeros(bs, G, f_z)

    for b in range(bs):
        for g in range(G):
            mask = gids[b] == g
            if not mask.any():
                continue
            prec = torch.exp(-agent_log_var[b, mask])
            sum_prec = prec.sum(dim=0).clamp(min=1e-8)
            group_mu[b, g] = (agent_mu[b, mask] * prec).sum(dim=0) / sum_prec
            group_log_var[b, g] = -torch.log(sum_prec)

    return group_mu, group_log_var


def _merge_ci_info_proportional_reference(
    agent_mu, agent_log_var, group_indices, device
):
    """Loop-based reference for CI fusion with info-proportional omega."""
    bs, n_agents, f_z = agent_mu.shape
    G = int(group_indices.max()) + 1
    gids = torch.as_tensor(group_indices, dtype=torch.long, device=device)

    group_mu = agent_mu.new_zeros(bs, G, f_z)
    group_log_var = agent_log_var.new_zeros(bs, G, f_z)

    for b in range(bs):
        for g in range(G):
            mask = gids[b] == g
            if not mask.any():
                continue
            p = torch.exp(-agent_log_var[b, mask])           # (n_g, L)
            omega = p / p.sum(dim=0, keepdim=True)           # info_proportional
            wp = omega * p                                   # (n_g, L)
            sum_wp = wp.sum(dim=0)
            group_log_var[b, g] = -torch.log(sum_wp)
            group_mu[b, g] = (wp * agent_mu[b, mask]).sum(dim=0) / sum_wp

    return group_mu, group_log_var


class TestGroupConsensusAgentGroup(unittest.TestCase):
    def setUp(self):
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
          in_features: 32
          out_features: 128
      decoder:
        model_type: "Custom"
        layers:
        - type: Linear
          in_features: 136
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
        self.agent_group_config = AgentGroupConfig(**config["agent_group"])
        self.agent_group = self.agent_group_config.get_agent_group()

    def test_agent_group_type(self):
        self.assertIsInstance(self.agent_group, GroupConsensusAgentGroup)

    def test_dead_agents_q_are_zero(self):
        bs = 2
        n_agents = 3
        obs_dim = 18
        seq_len = 5
        obs = torch.randn(bs, n_agents, seq_len, obs_dim)
        states = np.random.randn(bs, 1, 1)
        traj_padding_mask = torch.zeros(bs, seq_len)
        alive_mask = torch.ones(bs, n_agents, dtype=torch.bool)
        alive_mask[:, -1] = 0  # last agent is dead

        states_tensor = torch.from_numpy(states).float()
        ret = self.agent_group.forward(
            observations=obs, states=states_tensor, traj_padding_mask=traj_padding_mask,
            alive_mask=alive_mask,
        )
        q_val = ret["q_val"]
        dead_q = q_val[:, -1, :]
        self.assertTrue(torch.all(dead_q == 0), f"Dead agent Q-values should be zero, got {dead_q}")

    def test_dead_agents_group_indices_are_minus_one(self):
        bs = 2
        n_agents = 3
        obs_dim = 18
        seq_len = 5
        obs = torch.randn(bs, n_agents, seq_len, obs_dim)
        states = np.random.randn(bs, 1, 1)
        traj_padding_mask = torch.zeros(bs, seq_len)
        alive_mask = torch.ones(bs, n_agents, dtype=torch.bool)
        alive_mask[:, 0] = 0  # first agent is dead

        states_tensor = torch.from_numpy(states).float()
        ret = self.agent_group.forward(
            observations=obs, states=states_tensor, traj_padding_mask=traj_padding_mask,
            alive_mask=alive_mask,
        )
        group_indices = ret["group_indices"]
        self.assertEqual(group_indices.shape, (bs, n_agents))
        dead_gids = group_indices[:, 0]
        self.assertTrue(
            torch.all(dead_gids == -1),
            f"Dead agent group_indices should be -1, got {dead_gids}",
        )
        alive_gids = group_indices[:, 1:]
        self.assertTrue(
            torch.all(alive_gids >= 0),
            f"Alive agent group_indices should be >= 0, got {alive_gids}",
        )
    def test_group_consensus_registration(self):
        from marlite.algorithm.agents.agent_group_config import registered_agent_groups
        self.assertIn("GroupConsensusQMIX", registered_agent_groups)

    def test_ssl_group_consensus_mappo_registration(self):
        from marlite.algorithm.agents.agent_group_config import registered_agent_groups
        self.assertIn("SSLGroupConsensusMAPPO", registered_agent_groups)

    def test_group_consensus_mappo_registration(self):
        from marlite.algorithm.agents.agent_group_config import registered_agent_groups
        self.assertIn("GroupConsensusMAPPO", registered_agent_groups)


def _merge_group_mean_reference(agent_vectors, group_indices, device):
    bs, n_agents, f_z = agent_vectors.shape
    G = int(group_indices.max()) + 1
    gids = torch.as_tensor(group_indices, dtype=torch.long, device=device)

    group_mean = agent_vectors.new_zeros(bs, G, f_z)

    for b in range(bs):
        for g in range(G):
            mask = gids[b] == g
            if not mask.any():
                continue
            group_mean[b, g] = agent_vectors[b, mask].mean(dim=0)

    return group_mean, torch.zeros_like(group_mean)


class TestAEMerge(unittest.TestCase):
    def setUp(self):
        self.fake_self = FakeAgentGroup()
        self.bs, self.n_agents, self.f_z = 4, 6, 8

    def test_ae_mean_correctness(self):
        agent_vectors = torch.randn(self.bs, self.n_agents, self.f_z)
        group_indices = np.array([
            [0, 0, 1, 1, 2, 2],
            [0, 1, 2, 0, 1, 2],
            [0, 0, 0, 1, 1, 1],
            [2, 1, 0, 0, 1, 2],
        ], dtype=np.int16)
        group_indices = torch.from_numpy(group_indices)

        ref_mean, ref_lv = _merge_group_mean_reference(
            agent_vectors, group_indices, self.fake_self.device
        )
        new_mean, new_lv = GroupConsensusAgentGroup._merge_group_mean(
            self.fake_self, agent_vectors, group_indices
        )

        torch.testing.assert_close(new_mean, ref_mean, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(new_lv, ref_lv, atol=1e-5, rtol=1e-5)

    def test_ae_mean_with_dead_agents(self):
        agent_vectors = torch.randn(self.bs, self.n_agents, self.f_z)
        group_indices = np.array([
            [0, 0, -1, 1, 1, -1],
            [0, -1, -1, 1, 2, 2],
            [0, 0, 0, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
        ], dtype=np.int16)
        group_indices = torch.from_numpy(group_indices)

        ref_mean, ref_lv = _merge_group_mean_reference(
            agent_vectors, group_indices, self.fake_self.device
        )
        new_mean, new_lv = GroupConsensusAgentGroup._merge_group_mean(
            self.fake_self, agent_vectors, group_indices
        )

        torch.testing.assert_close(new_mean, ref_mean, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(new_lv, ref_lv, atol=1e-5, rtol=1e-5)

    def test_ae_mean_no_nan_inf(self):
        agent_vectors = torch.randn(2, self.n_agents, self.f_z)
        group_indices = np.array([
            [0, 0, 1, 1, 2, 2],
            [0, 0, 0, 1, 1, 1],
        ], dtype=np.int16)
        group_indices = torch.from_numpy(group_indices)

        new_mean, new_lv = GroupConsensusAgentGroup._merge_group_mean(
            self.fake_self, agent_vectors, group_indices
        )

        self.assertFalse(torch.isnan(new_mean).any())
        self.assertFalse(torch.isnan(new_lv).any())
        self.assertFalse(torch.isinf(new_mean).any())
        self.assertFalse(torch.isinf(new_lv).any())

    def test_ae_mean_backward(self):
        agent_vectors = torch.randn(
            1, self.n_agents, self.f_z, requires_grad=True
        )
        group_indices = np.array([[0, 0, 1, 1, 2, 2]], dtype=np.int16)
        group_indices = torch.from_numpy(group_indices)

        new_mean, new_lv = GroupConsensusAgentGroup._merge_group_mean(
            self.fake_self, agent_vectors, group_indices
        )
        loss = new_mean.sum()
        loss.backward()

        self.assertIsNotNone(agent_vectors.grad)
        self.assertFalse(torch.isnan(agent_vectors.grad).any())
        self.assertFalse(torch.isinf(agent_vectors.grad).any())

    def test_ae_mean_identical_agents(self):
        group_indices = np.array([[0, 0, 1, 1, 2, 2]], dtype=np.int16)
        group_indices = torch.from_numpy(group_indices)
        agent_vectors = torch.ones(1, self.n_agents, self.f_z) * 5.0

        new_mean, new_lv = GroupConsensusAgentGroup._merge_group_mean(
            self.fake_self, agent_vectors, group_indices
        )

        self.assertTrue((new_mean[0, 0].abs().sub(5.0).abs().max() < 1e-5).item())
        self.assertTrue(torch.all(new_lv == 0))


class TestConsensusModeDispatch(unittest.TestCase):
    def setUp(self):
        self.fake = FakeAgentGroup()
        self.bs, self.n_agents, self.f_z = 4, 6, 8

    def test_ae_dispatch_matches_merge_group_mean(self):
        self.fake.merge_mode = "bayesian"
        agent_vectors = torch.randn(self.bs, self.n_agents, self.f_z)
        group_indices = np.array([
            [0, 0, 1, 1, 2, 2],
            [0, 1, 2, 0, 1, 2],
            [0, 0, 0, 1, 1, 1],
            [2, 1, 0, 0, 1, 2],
        ], dtype=np.int16)
        group_indices = torch.from_numpy(group_indices)

        ae_mean, ae_lv = GroupConsensusAgentGroup._merge_group_mean(
            self.fake, agent_vectors, group_indices
        )

        self.assertTrue(torch.all(ae_lv == 0))

        for b in range(self.bs):
            for g in range(int(group_indices[b].max()) + 1):
                mask = group_indices[b] == g
                if mask.any():
                    expected = agent_vectors[b, mask].mean(dim=0)
                    torch.testing.assert_close(
                        ae_mean[b, g], expected, atol=1e-5, rtol=1e-5
                    )

    def test_ae_mean_same_as_sample_mean_mu(self):
        agent_vectors = torch.randn(self.bs, self.n_agents, self.f_z)
        agent_log_var = torch.randn(self.bs, self.n_agents, self.f_z) * 3.0 - 1.0
        group_indices = np.array([
            [0, 0, 1, 1, 2, 2],
            [0, 1, 2, 0, 1, 2],
            [0, 0, 0, 1, 1, 1],
            [2, 1, 0, 0, 1, 2],
        ], dtype=np.int16)
        group_indices = torch.from_numpy(group_indices)

        fake_sm = FakeAgentGroup()
        fake_sm.merge_mode = "sample_mean"

        ae_mean, ae_lv = GroupConsensusAgentGroup._merge_group_mean(
            fake_sm, agent_vectors, group_indices
        )
        sm_mean, sm_lv = GroupConsensusAgentGroup._merge_sample_mean(
            fake_sm, agent_vectors, agent_log_var, group_indices
        )

        self.assertTrue(torch.allclose(ae_mean, sm_mean, atol=1e-5, rtol=1e-5),
                         "AE mean should produce same mu as sample_mean")
        self.assertFalse(torch.allclose(ae_lv, sm_lv),
                         "AE log_var (zeros) and sample_mean log_var should differ")
        self.assertTrue(torch.all(ae_lv == 0), "AE log_var should be all zeros")

    def test_ae_differs_from_bayesian(self):
        agent_vectors = torch.randn(self.bs, self.n_agents, self.f_z)
        agent_log_var = torch.randn(self.bs, self.n_agents, self.f_z) * 3.0 - 1.0
        group_indices = np.array([
            [0, 0, 1, 1, 2, 2],
            [0, 1, 2, 0, 1, 2],
            [0, 0, 0, 1, 1, 1],
            [2, 1, 0, 0, 1, 2],
        ], dtype=np.int16)
        group_indices = torch.from_numpy(group_indices)

        fake_bs = FakeAgentGroup()
        fake_bs.merge_mode = "bayesian"

        ae_mean, ae_lv = GroupConsensusAgentGroup._merge_group_mean(
            fake_bs, agent_vectors, group_indices
        )
        bay_mean, bay_lv = GroupConsensusAgentGroup._merge_bayesian(
            fake_bs, agent_vectors, agent_log_var, group_indices
        )

        self.assertFalse(torch.allclose(ae_mean, bay_mean),
                         "AE mean and bayesian should produce different mu")
        self.assertFalse(torch.allclose(ae_lv, bay_lv),
                         "AE log_var (zeros) and bayesian log_var should differ")


class TestBayesianMerge(unittest.TestCase):
    def setUp(self):
        self.fake_self = FakeAgentGroup()
        self.bs, self.n_agents, self.f_z = 4, 6, 8

    def _run_comparison(self, group_indices_np):
        agent_mu = torch.randn(self.bs, self.n_agents, self.f_z)
        agent_log_var = torch.randn(self.bs, self.n_agents, self.f_z) * 2.0 - 2.0

        ref_mu, ref_lv = _merge_bayesian_reference(
            agent_mu, agent_log_var, group_indices_np, self.fake_self.device
        )
        new_mu, new_lv = GroupConsensusAgentGroup._merge_bayesian(
            self.fake_self, agent_mu, agent_log_var, group_indices_np
        )

        torch.testing.assert_close(ref_mu, new_mu, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(ref_lv, new_lv, atol=1e-5, rtol=1e-5)

    def test_bayesian_normal_grouping(self):
        group_indices = np.array([
            [0, 0, 1, 1, 2, 2],
            [0, 1, 2, 0, 1, 2],
            [0, 0, 0, 1, 1, 1],
            [2, 1, 0, 0, 1, 2],
        ], dtype=np.int16)
        group_indices = torch.from_numpy(group_indices)
        self._run_comparison(group_indices)

    def test_bayesian_with_dead_agents(self):
        group_indices = np.array([
            [0, 0, -1, 1, 1, -1],
            [0, -1, -1, 1, 2, 2],
            [0, 0, 0, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
        ], dtype=np.int16)
        group_indices = torch.from_numpy(group_indices)
        self._run_comparison(group_indices)

    def test_bayesian_single_group(self):
        group_indices = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ], dtype=np.int16)
        group_indices = torch.from_numpy(group_indices)
        self._run_comparison(group_indices)

    def test_bayesian_each_agent_own_group(self):
        group_indices = np.array([
            [0, 1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1, 0],
            [0, 1, 2, 3, 4, 5],
            [2, 3, 0, 1, 5, 4],
        ], dtype=np.int16)
        group_indices = torch.from_numpy(group_indices)
        self._run_comparison(group_indices)

    def test_bayesian_identical_agents(self):
        group_indices = np.array([
            [0, 0, 1, 1, 2, 2],
        ], dtype=np.int16)
        group_indices = torch.from_numpy(group_indices)
        agent_mu = torch.ones(1, self.n_agents, self.f_z) * 3.0
        agent_log_var = torch.ones(1, self.n_agents, self.f_z) * -1.0

        ref_mu, ref_lv = _merge_bayesian_reference(
            agent_mu, agent_log_var, group_indices, self.fake_self.device
        )
        new_mu, new_lv = GroupConsensusAgentGroup._merge_bayesian(
            self.fake_self, agent_mu, agent_log_var, group_indices
        )
        torch.testing.assert_close(ref_mu, new_mu, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(ref_lv, new_lv, atol=1e-5, rtol=1e-5)

        self.assertTrue((new_mu[0, 0].abs().sub(3.0).abs().max() < 1e-5).item())

    def test_bayesian_low_variance_has_more_weight(self):
        group_indices = np.array([[0, 0]], dtype=np.int16)
        group_indices = torch.from_numpy(group_indices)
        agent_mu = torch.tensor([[[1.0, 2.0, 3.0], [8.0, 9.0, 10.0]]])
        agent_log_var = torch.tensor([[[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]]])

        new_mu, new_lv = GroupConsensusAgentGroup._merge_bayesian(
            self.fake_self, agent_mu, agent_log_var, group_indices
        )

        dist_to_agent0 = abs(new_mu[0, 0, 0].item() - 1.0)
        dist_to_agent1 = abs(new_mu[0, 0, 0].item() - 8.0)
        self.assertLess(dist_to_agent0, dist_to_agent1,
                        "Low-variance agent (mu=1) should have more weight than high-variance agent (mu=8)")

    def test_bayesian_output_no_nan_inf(self):
        group_indices = np.array([
            [0, 0, 1, 1, 2, 2],
            [0, 0, 0, 1, 1, 1],
        ], dtype=np.int16)
        group_indices = torch.from_numpy(group_indices)
        agent_mu = torch.randn(2, self.n_agents, self.f_z)
        agent_log_var = torch.randn(2, self.n_agents, self.f_z) - 1.0

        new_mu, new_lv = GroupConsensusAgentGroup._merge_bayesian(
            self.fake_self, agent_mu, agent_log_var, group_indices
        )

        self.assertFalse(torch.isnan(new_mu).any())
        self.assertFalse(torch.isnan(new_lv).any())
        self.assertFalse(torch.isinf(new_mu).any())
        self.assertFalse(torch.isinf(new_lv).any())

    def test_bayesian_backward(self):
        group_indices = np.array([
            [0, 0, 1, 1, 2, 2],
        ], dtype=np.int16)
        group_indices = torch.from_numpy(group_indices)
        agent_mu = torch.randn(1, self.n_agents, self.f_z, requires_grad=True)
        agent_log_var = torch.randn(1, self.n_agents, self.f_z, requires_grad=True)

        new_mu, new_lv = GroupConsensusAgentGroup._merge_bayesian(
            self.fake_self, agent_mu, agent_log_var, group_indices
        )
        loss = new_mu.sum() + new_lv.sum()
        loss.backward()

        self.assertIsNotNone(agent_mu.grad)
        self.assertIsNotNone(agent_log_var.grad)
        self.assertFalse(torch.isnan(agent_mu.grad).any())
        self.assertFalse(torch.isnan(agent_log_var.grad).any())
        self.assertFalse(torch.isinf(agent_mu.grad).any())
        self.assertFalse(torch.isinf(agent_log_var.grad).any())

    def test_bayesian_large_batch(self):
        bs, n_agents, f_z = 64, 36, 64
        agent_mu = torch.randn(bs, n_agents, f_z)
        agent_log_var = torch.randn(bs, n_agents, f_z) - 1.0
        group_indices = np.random.randint(0, 6, size=(bs, n_agents)).astype(np.int16)
        group_indices = torch.from_numpy(group_indices)
        group_indices[group_indices == 5] = -1

        ref_mu, ref_lv = _merge_bayesian_reference(
            agent_mu, agent_log_var, group_indices, self.fake_self.device
        )
        new_mu, new_lv = GroupConsensusAgentGroup._merge_bayesian(
            self.fake_self, agent_mu, agent_log_var, group_indices
        )

        torch.testing.assert_close(ref_mu, new_mu, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(ref_lv, new_lv, atol=1e-5, rtol=1e-5)


class TestCIMerge(unittest.TestCase):
    def setUp(self):
        self.fake_self = FakeAgentGroup()
        self.fake_self.ci_omega_mode = "info_proportional"
        self.bs, self.n_agents, self.f_z = 4, 6, 8

    def _run_comparison(self, group_indices_np):
        agent_mu = torch.randn(self.bs, self.n_agents, self.f_z)
        agent_log_var = torch.randn(self.bs, self.n_agents, self.f_z) * 2.0 - 2.0

        ref_mu, ref_lv = _merge_ci_info_proportional_reference(
            agent_mu, agent_log_var, group_indices_np, self.fake_self.device
        )
        new_mu, new_lv = GroupConsensusAgentGroup._merge_ci(
            self.fake_self, agent_mu, agent_log_var, group_indices_np
        )

        torch.testing.assert_close(ref_mu, new_mu, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(ref_lv, new_lv, atol=1e-5, rtol=1e-5)

    def test_ci_normal_grouping(self):
        group_indices = np.array([
            [0, 0, 1, 1, 2, 2],
            [0, 1, 2, 0, 1, 2],
            [0, 0, 0, 1, 1, 1],
            [2, 1, 0, 0, 1, 2],
        ], dtype=np.int16)
        group_indices = torch.from_numpy(group_indices)
        self._run_comparison(group_indices)

    def test_ci_with_dead_agents(self):
        group_indices = np.array([
            [0, 0, -1, 1, 1, -1],
            [0, -1, -1, 1, 2, 2],
            [0, 0, 0, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
        ], dtype=np.int16)
        group_indices = torch.from_numpy(group_indices)
        self._run_comparison(group_indices)

    def test_ci_single_group(self):
        group_indices = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ], dtype=np.int16)
        group_indices = torch.from_numpy(group_indices)
        self._run_comparison(group_indices)

    def test_ci_each_agent_own_group(self):
        group_indices = np.array([
            [0, 1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1, 0],
            [0, 1, 2, 3, 4, 5],
            [2, 3, 0, 1, 5, 4],
        ], dtype=np.int16)
        group_indices = torch.from_numpy(group_indices)
        self._run_comparison(group_indices)

    def test_ci_large_batch(self):
        bs, n_agents, f_z = 64, 36, 64
        agent_mu = torch.randn(bs, n_agents, f_z)
        agent_log_var = torch.randn(bs, n_agents, f_z) - 1.0
        group_indices = np.random.randint(0, 6, size=(bs, n_agents)).astype(np.int16)
        group_indices = torch.from_numpy(group_indices)
        group_indices[group_indices == 5] = -1

        ref_mu, ref_lv = _merge_ci_info_proportional_reference(
            agent_mu, agent_log_var, group_indices, self.fake_self.device
        )
        new_mu, new_lv = GroupConsensusAgentGroup._merge_ci(
            self.fake_self, agent_mu, agent_log_var, group_indices
        )

        torch.testing.assert_close(ref_mu, new_mu, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(ref_lv, new_lv, atol=1e-5, rtol=1e-5)

    def test_ci_single_member_identity(self):
        group_indices = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int16)
        group_indices = torch.from_numpy(group_indices)
        agent_mu = torch.randn(1, self.n_agents, self.f_z)
        agent_log_var = torch.randn(1, self.n_agents, self.f_z) * 2.0 - 2.0

        new_mu, new_lv = GroupConsensusAgentGroup._merge_ci(
            self.fake_self, agent_mu, agent_log_var, group_indices
        )

        torch.testing.assert_close(new_mu, agent_mu, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(new_lv, agent_log_var, atol=1e-6, rtol=1e-6)

    def test_ci_variance_lower_bound(self):
        group_indices = np.array([
            [0, 0, 1, 1, 2, 2],
            [0, 0, 0, 1, 1, 1],
            [2, 1, 0, 0, 1, 2],
            [0, 0, 0, 0, 0, 0],
        ], dtype=np.int16)
        group_indices = torch.from_numpy(group_indices)
        agent_mu = torch.randn(self.bs, self.n_agents, self.f_z)
        agent_log_var = torch.randn(self.bs, self.n_agents, self.f_z) * 2.0 - 2.0

        ci_mu, ci_lv = GroupConsensusAgentGroup._merge_ci(
            self.fake_self, agent_mu, agent_log_var, group_indices
        )
        bay_mu, bay_lv = GroupConsensusAgentGroup._merge_bayesian(
            self.fake_self, agent_mu, agent_log_var, group_indices
        )

        gids = group_indices.to(dtype=torch.long)
        dead = gids < 0
        gids_safe = gids.clamp(min=0)
        mask = F.one_hot(gids_safe, num_classes=ci_lv.shape[1]).float()
        mask[dead] = 0.0

        neg_inf = torch.tensor(float('-inf'))
        agent_lv_exp = agent_log_var.unsqueeze(2)
        mask_z = mask.unsqueeze(-1)
        min_member_lv, _ = torch.where(
            mask_z.bool(), agent_lv_exp, neg_inf
        ).min(dim=1)

        # CI variance must never fall below the most confident member.
        self.assertTrue(
            torch.all(ci_lv >= min_member_lv - 1e-5),
            "CI fused variance must be >= min member variance (consistency)",
        )
        # CI is more conservative than the independent-Bayes fusion.
        self.assertTrue(
            torch.all(ci_lv >= bay_lv - 1e-5),
            "CI fused variance must be >= Bayesian fused variance",
        )

    def test_ci_precision_squared_weighting(self):
        group_indices = np.array([[0, 0]], dtype=np.int16)
        group_indices = torch.from_numpy(group_indices)
        agent_mu = torch.tensor([[[1.0, 2.0, 3.0], [8.0, 9.0, 10.0]]])
        agent_log_var = torch.tensor([[[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]]])

        ci_mu, _ = GroupConsensusAgentGroup._merge_ci(
            self.fake_self, agent_mu, agent_log_var, group_indices
        )
        bay_mu, _ = GroupConsensusAgentGroup._merge_bayesian(
            self.fake_self, agent_mu, agent_log_var, group_indices
        )

        dist_ci = (ci_mu[0, 0, 0].item() - 1.0) ** 2
        dist_bay = (bay_mu[0, 0, 0].item() - 1.0) ** 2
        self.assertLess(
            dist_ci, dist_bay,
            "CI info-proportional weights should favor the low-variance agent "
            "more strongly than Bayesian precision weighting",
        )

    def test_ci_output_no_nan_inf(self):
        group_indices = np.array([
            [0, 0, 1, 1, 2, 2],
            [0, 0, 0, 1, 1, 1],
        ], dtype=np.int16)
        group_indices = torch.from_numpy(group_indices)
        agent_mu = torch.randn(2, self.n_agents, self.f_z)
        agent_log_var = torch.randn(2, self.n_agents, self.f_z) * 2.0 - 2.0

        new_mu, new_lv = GroupConsensusAgentGroup._merge_ci(
            self.fake_self, agent_mu, agent_log_var, group_indices
        )

        self.assertFalse(torch.isnan(new_mu).any())
        self.assertFalse(torch.isnan(new_lv).any())
        self.assertFalse(torch.isinf(new_mu).any())
        self.assertFalse(torch.isinf(new_lv).any())

    def test_ci_backward(self):
        group_indices = np.array([
            [0, 0, 1, 1, 2, 2],
        ], dtype=np.int16)
        group_indices = torch.from_numpy(group_indices)
        agent_mu = torch.randn(1, self.n_agents, self.f_z, requires_grad=True)
        agent_log_var = torch.randn(
            1, self.n_agents, self.f_z, requires_grad=True
        )

        new_mu, new_lv = GroupConsensusAgentGroup._merge_ci(
            self.fake_self, agent_mu, agent_log_var, group_indices
        )
        loss = new_mu.sum() + new_lv.sum()
        loss.backward()

        self.assertIsNotNone(agent_mu.grad)
        self.assertIsNotNone(agent_log_var.grad)
        self.assertFalse(torch.isnan(agent_mu.grad).any())
        self.assertFalse(torch.isnan(agent_log_var.grad).any())
        self.assertFalse(torch.isinf(agent_mu.grad).any())
        self.assertFalse(torch.isinf(agent_log_var.grad).any())

    def test_ci_dispatch_matches_direct_call(self):
        self.fake_self.merge_mode = "ci"
        agent_mu = torch.randn(self.bs, self.n_agents, self.f_z)
        agent_log_var = torch.randn(self.bs, self.n_agents, self.f_z) * 2.0 - 2.0
        group_indices = np.array([
            [0, 0, 1, 1, 2, 2],
            [0, 1, 2, 0, 1, 2],
            [0, 0, 0, 1, 1, 1],
            [2, 1, 0, 0, 1, 2],
        ], dtype=np.int16)
        group_indices = torch.from_numpy(group_indices)

        result_mu, result_lv = GroupConsensusAgentGroup._merge_group_distributions(
            self.fake_self, agent_mu, agent_log_var, group_indices
        )
        expected_mu, expected_lv = GroupConsensusAgentGroup._merge_ci(
            self.fake_self, agent_mu, agent_log_var, group_indices
        )

        torch.testing.assert_close(result_mu, expected_mu)
        torch.testing.assert_close(result_lv, expected_lv)


class TestMergeModeDispatch(unittest.TestCase):
    def setUp(self):
        self.fake = FakeAgentGroup()
        self.bs, self.n_agents, self.f_z = 4, 6, 8

    def test_dispatch_sample_mean(self):
        self.fake.merge_mode = "sample_mean"
        agent_mu = torch.randn(self.bs, self.n_agents, self.f_z)
        agent_log_var = torch.randn(self.bs, self.n_agents, self.f_z) - 2.0
        group_indices = np.array([
            [0, 0, 1, 1, 2, 2],
            [0, 1, 2, 0, 1, 2],
            [0, 0, 0, 1, 1, 1],
            [2, 1, 0, 0, 1, 2],
        ], dtype=np.int16)
        group_indices = torch.from_numpy(group_indices)

        result_mu, result_lv = GroupConsensusAgentGroup._merge_group_distributions(
            self.fake, agent_mu, agent_log_var, group_indices
        )
        expected_mu, expected_lv = GroupConsensusAgentGroup._merge_sample_mean(
            self.fake, agent_mu, agent_log_var, group_indices
        )

        torch.testing.assert_close(result_mu, expected_mu)
        torch.testing.assert_close(result_lv, expected_lv)

    def test_dispatch_bayesian(self):
        self.fake.merge_mode = "bayesian"
        agent_mu = torch.randn(self.bs, self.n_agents, self.f_z)
        agent_log_var = torch.randn(self.bs, self.n_agents, self.f_z) - 2.0
        group_indices = np.array([
            [0, 0, 1, 1, 2, 2],
            [0, 1, 2, 0, 1, 2],
            [0, 0, 0, 1, 1, 1],
            [2, 1, 0, 0, 1, 2],
        ], dtype=np.int16)
        group_indices = torch.from_numpy(group_indices)

        result_mu, result_lv = GroupConsensusAgentGroup._merge_group_distributions(
            self.fake, agent_mu, agent_log_var, group_indices
        )
        expected_mu, expected_lv = GroupConsensusAgentGroup._merge_bayesian(
            self.fake, agent_mu, agent_log_var, group_indices
        )

        torch.testing.assert_close(result_mu, expected_mu)
        torch.testing.assert_close(result_lv, expected_lv)

    def test_two_modes_produce_different_results(self):
        agent_mu = torch.randn(self.bs, self.n_agents, self.f_z)
        agent_log_var = torch.randn(self.bs, self.n_agents, self.f_z) * 3.0 - 1.0
        group_indices = np.array([
            [0, 0, 1, 1, 2, 2],
            [0, 1, 2, 0, 1, 2],
            [0, 0, 0, 1, 1, 1],
            [2, 1, 0, 0, 1, 2],
        ], dtype=np.int16)
        group_indices = torch.from_numpy(group_indices)

        fake_sm = FakeAgentGroup()
        fake_sm.merge_mode = "sample_mean"
        fake_bs = FakeAgentGroup()
        fake_bs.merge_mode = "bayesian"

        mu_sm, lv_sm = GroupConsensusAgentGroup._merge_group_distributions(
            fake_sm, agent_mu, agent_log_var, group_indices
        )
        mu_bs, lv_bs = GroupConsensusAgentGroup._merge_group_distributions(
            fake_bs, agent_mu, agent_log_var, group_indices
        )

        self.assertFalse(torch.allclose(mu_sm, mu_bs),
                         "Bayesian and sample_mean should produce different mu")
        self.assertFalse(torch.allclose(lv_sm, lv_bs),
                         "Bayesian and sample_mean should produce different log_var")


class TestScatterRoundtrip(unittest.TestCase):
    def test_roundtrip_sample_mean(self):
        bs, n_agents, f_z = 4, 6, 8
        fake = FakeAgentGroup()
        fake.merge_mode = "sample_mean"
        agent_mu = torch.randn(bs, n_agents, f_z)
        agent_log_var = torch.randn(bs, n_agents, f_z) - 1.0
        group_indices = np.array([
            [0, 0, 1, 1, 2, 2],
            [0, 0, 0, 1, 1, 1],
            [2, 1, 0, 0, 1, 2],
            [0, 0, 0, 0, 0, 0],
        ], dtype=np.int16)
        group_indices = torch.from_numpy(group_indices)

        group_mu, group_log_var = GroupConsensusAgentGroup._merge_group_distributions(
            fake, agent_mu, agent_log_var, group_indices
        )
        scattered_mu = GroupConsensusAgentGroup._scatter(group_mu, group_indices)

        for b in range(bs):
            for i in range(n_agents):
                gid = group_indices[b, i]
                if gid >= 0:
                    self.assertTrue(torch.allclose(scattered_mu[b, i], group_mu[b, gid]))

    def test_roundtrip_bayesian(self):
        bs, n_agents, f_z = 4, 6, 8
        fake = FakeAgentGroup()
        fake.merge_mode = "bayesian"
        agent_mu = torch.randn(bs, n_agents, f_z)
        agent_log_var = torch.randn(bs, n_agents, f_z) - 1.0
        group_indices = np.array([
            [0, 0, 1, 1, 2, 2],
            [0, 0, 0, 1, 1, 1],
            [2, 1, 0, 0, 1, 2],
            [0, 0, 0, 0, 0, 0],
        ], dtype=np.int16)
        group_indices = torch.from_numpy(group_indices)

        group_mu, group_log_var = GroupConsensusAgentGroup._merge_group_distributions(
            fake, agent_mu, agent_log_var, group_indices
        )
        scattered_mu = GroupConsensusAgentGroup._scatter(group_mu, group_indices)

        for b in range(bs):
            for i in range(n_agents):
                gid = group_indices[b, i]
                if gid >= 0:
                    self.assertTrue(torch.allclose(scattered_mu[b, i], group_mu[b, gid]))

    def test_roundtrip_ci(self):
        bs, n_agents, f_z = 4, 6, 8
        fake = FakeAgentGroup()
        fake.merge_mode = "ci"
        fake.ci_omega_mode = "info_proportional"
        agent_mu = torch.randn(bs, n_agents, f_z)
        agent_log_var = torch.randn(bs, n_agents, f_z) - 1.0
        group_indices = np.array([
            [0, 0, 1, 1, 2, 2],
            [0, 0, 0, 1, 1, 1],
            [2, 1, 0, 0, 1, 2],
            [0, 0, 0, 0, 0, 0],
        ], dtype=np.int16)
        group_indices = torch.from_numpy(group_indices)

        group_mu, group_log_var = GroupConsensusAgentGroup._merge_group_distributions(
            fake, agent_mu, agent_log_var, group_indices
        )
        scattered_mu = GroupConsensusAgentGroup._scatter(group_mu, group_indices)

        for b in range(bs):
            for i in range(n_agents):
                gid = group_indices[b, i]
                if gid >= 0:
                    self.assertTrue(torch.allclose(scattered_mu[b, i], group_mu[b, gid]))


class TestCIConstruction(unittest.TestCase):
    _BASE_CONFIG = """
agent_group:
  type: "GroupConsensusQMIX"
  merge_mode: "ci"
  ci_omega_mode: "info_proportional"
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
          in_features: 32
          out_features: 128
      decoder:
        model_type: "Custom"
        layers:
        - type: Linear
          in_features: 136
          out_features: 5
  group_builder:
    type: "Fixed"
    group_ids: [0, 0, 1]
"""

    def test_ci_config_constructs(self):
        config = yaml.safe_load(self._BASE_CONFIG)
        agent_group = AgentGroupConfig(**config["agent_group"]).get_agent_group()
        self.assertIsInstance(agent_group, GroupConsensusAgentGroup)
        self.assertEqual(agent_group.merge_mode, "ci")
        self.assertEqual(agent_group.ci_omega_mode, "info_proportional")

    def test_ci_forward_runs(self):
        config = yaml.safe_load(self._BASE_CONFIG)
        agent_group = AgentGroupConfig(**config["agent_group"]).get_agent_group()
        bs, n_agents, obs_dim, seq_len = 2, 3, 18, 5
        obs = torch.randn(bs, n_agents, seq_len, obs_dim)
        states = np.random.randn(bs, 1, 1)
        traj_padding_mask = torch.zeros(bs, seq_len)
        alive_mask = torch.ones(bs, n_agents, dtype=torch.bool)

        ret = agent_group.forward(
            observations=obs,
            states=torch.from_numpy(states).float(),
            traj_padding_mask=traj_padding_mask,
            alive_mask=alive_mask,
        )
        self.assertEqual(ret["q_val"].shape, (bs, n_agents, 5))
        self.assertFalse(torch.isnan(ret["q_val"]).any())
        self.assertFalse(torch.isinf(ret["q_val"]).any())

    def test_ci_default_omega_mode(self):
        config = yaml.safe_load(
            self._BASE_CONFIG.replace('  ci_omega_mode: "info_proportional"\n', "")
        )
        agent_group = AgentGroupConfig(**config["agent_group"]).get_agent_group()
        self.assertEqual(agent_group.ci_omega_mode, "info_proportional")

    def test_unsupported_merge_mode_raises(self):
        config = yaml.safe_load(
            self._BASE_CONFIG.replace('merge_mode: "ci"', 'merge_mode: "nonexistent"')
        )
        with self.assertRaises(ValueError):
            AgentGroupConfig(**config["agent_group"]).get_agent_group()

    def test_unsupported_ci_omega_mode_raises(self):
        config = yaml.safe_load(
            self._BASE_CONFIG.replace(
                'ci_omega_mode: "info_proportional"',
                'ci_omega_mode: "learnable"',
            )
        )
        with self.assertRaises(ValueError):
            AgentGroupConfig(**config["agent_group"]).get_agent_group()


if __name__ == "__main__":
    unittest.main()
