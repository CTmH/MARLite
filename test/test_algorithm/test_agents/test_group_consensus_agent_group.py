import os
import unittest
import numpy as np
import yaml
import torch

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


class TestGroupConsensusAgentGroup(unittest.TestCase):
    def setUp(self):
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "config", "group_consensus_default.yaml"
        )
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.agent_group_config = AgentGroupConfig(**config["agent_group"])
        self.agent_group = self.agent_group_config.get_agent_group()

    def test_agent_group_type(self):
        self.assertIsInstance(self.agent_group, GroupConsensusAgentGroup)


class TestGroupConsensusAgentGroupConfig(unittest.TestCase):
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
        self._run_comparison(group_indices)

    def test_bayesian_with_dead_agents(self):
        group_indices = np.array([
            [0, 0, -1, 1, 1, -1],
            [0, -1, -1, 1, 2, 2],
            [0, 0, 0, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
        ], dtype=np.int16)
        self._run_comparison(group_indices)

    def test_bayesian_single_group(self):
        group_indices = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ], dtype=np.int16)
        self._run_comparison(group_indices)

    def test_bayesian_each_agent_own_group(self):
        group_indices = np.array([
            [0, 1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1, 0],
            [0, 1, 2, 3, 4, 5],
            [2, 3, 0, 1, 5, 4],
        ], dtype=np.int16)
        self._run_comparison(group_indices)

    def test_bayesian_identical_agents(self):
        group_indices = np.array([
            [0, 0, 1, 1, 2, 2],
        ], dtype=np.int16)
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
        group_indices[group_indices == 5] = -1

        ref_mu, ref_lv = _merge_bayesian_reference(
            agent_mu, agent_log_var, group_indices, self.fake_self.device
        )
        new_mu, new_lv = GroupConsensusAgentGroup._merge_bayesian(
            self.fake_self, agent_mu, agent_log_var, group_indices
        )

        torch.testing.assert_close(ref_mu, new_mu, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(ref_lv, new_lv, atol=1e-5, rtol=1e-5)


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

        group_mu, group_log_var = GroupConsensusAgentGroup._merge_group_distributions(
            fake, agent_mu, agent_log_var, group_indices
        )
        scattered_mu = GroupConsensusAgentGroup._scatter(group_mu, group_indices)

        for b in range(bs):
            for i in range(n_agents):
                gid = group_indices[b, i]
                if gid >= 0:
                    self.assertTrue(torch.allclose(scattered_mu[b, i], group_mu[b, gid]))


if __name__ == "__main__":
    unittest.main()
