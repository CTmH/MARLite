"""Tests for all Mixer subclasses (QMixer, SeqQMixer, ProbQMixer, ProbSeqQMixer, GroupConsensusMixer).

Each test builds a config dict (no YAML), creates the critic through the
CriticConfig factory, and runs a forward pass with dummy data to verify
output shapes and differentiability.
"""

import torch
import unittest

from marlite.algorithm.critic import QMixer, SeqQMixer, ProbQMixer, ProbSeqQMixer
from marlite.algorithm.critic.group_consensus_mixer import GroupConsensusMixer
from marlite.algorithm.critic.critic_config import CriticConfig, registered_critic_creators


B = 4
T = 5
N = 3
F = 18
STATE_FLAT = N * F  # 54


def _mpe_state():
    """Flat state as returned by MPE env.state(): (B, T, N*F)."""
    return torch.randn(B, T, STATE_FLAT)


def _mpe_alive():
    """Alive mask as used in QMIXTrainer: (B, T, N)."""
    return torch.ones(B, T, N, dtype=torch.bool)


def _pad():
    """Padding mask: (B, T)."""
    return torch.zeros(B, T, dtype=torch.bool)


class TestMixerRegistration(unittest.TestCase):
    """All five Mixer subclasses must be in the factory registry."""

    def test_all_registered(self):
        for key in ("QMixer", "SeqQMixer", "ProbQMixer",
                     "ProbSeqQMixer", "GroupConsensusMixer"):
            self.assertIn(key, registered_critic_creators)


class TestQMixer(unittest.TestCase):
    """Standard QMixer — last-timestep mixing."""

    def setUp(self):
        cfg = {
            "type": "QMixer",
            "model": {
                "model_type": "QMixModel",
                "state_shape": STATE_FLAT,
                "input_dim": N,
                "qmix_hidden_dim": 16,
            },
            "feature_extractor": {"model_type": "Identity"},
        }
        self.critic = CriticConfig(**cfg).get_critic()
        self.assertIsInstance(self.critic, QMixer)

    def test_forward_shape(self):
        out = self.critic(torch.randn(B, N), _mpe_state(), _mpe_alive(), _pad())
        self.assertEqual(out["q_tot"].shape, (B,))
        self.assertIn("state_features", out)

    def test_backward(self):
        q = torch.randn(B, N, requires_grad=True)
        out = self.critic(q, _mpe_state(), _mpe_alive(), _pad())
        out["q_tot"].sum().backward()
        self.assertIsNotNone(q.grad)


class TestSeqQMixer(unittest.TestCase):
    """Sequential QMixer — full sequence through seq_model."""

    def setUp(self):
        cfg = {
            "type": "SeqQMixer",
            "model": {
                "model_type": "QMixModel",
                "state_shape": 32,
                "input_dim": N,
                "qmix_hidden_dim": 16,
            },
            "feature_extractor": {"model_type": "Identity"},
            "seq_model": {
                "model_type": "Custom",
                "layers": [
                    {"type": "Linear", "in_features": STATE_FLAT, "out_features": 32},
                    {"type": "ReLU"},
                ],
            },
        }
        self.critic = CriticConfig(**cfg).get_critic()
        self.assertIsInstance(self.critic, SeqQMixer)

    def test_forward_shape(self):
        out = self.critic(torch.randn(B, N), _mpe_state(), _mpe_alive(), _pad())
        self.assertEqual(out["q_tot"].shape, (B,))

    def test_backward(self):
        q = torch.randn(B, N, requires_grad=True)
        out = self.critic(q, _mpe_state(), _mpe_alive(), _pad())
        out["q_tot"].sum().backward()
        self.assertIsNotNone(q.grad)


class TestProbQMixer(unittest.TestCase):
    """Probabilistic QMixer — variational state features."""

    def setUp(self):
        cfg = {
            "type": "ProbQMixer",
            "model": {
                "model_type": "QMixModel",
                "state_shape": 16,
                "input_dim": N,
                "qmix_hidden_dim": 16,
            },
            "feature_extractor": {
                "model_type": "Custom",
                "layers": [
                    {"type": "Linear", "in_features": STATE_FLAT, "out_features": 32},
                ],
            },
            "deterministic_eval": False,
        }
        self.critic = CriticConfig(**cfg).get_critic()
        self.assertIsInstance(self.critic, ProbQMixer)

    def test_forward_shape(self):
        out = self.critic(torch.randn(B, N), _mpe_state(), _mpe_alive(), _pad())
        self.assertEqual(out["q_tot"].shape, (B,))
        self.assertIn("mu", out)
        self.assertIn("std", out)

    def test_backward(self):
        q = torch.randn(B, N, requires_grad=True)
        out = self.critic(q, _mpe_state(), _mpe_alive(), _pad())
        out["q_tot"].sum().backward()
        self.assertIsNotNone(q.grad)


class TestProbSeqQMixer(unittest.TestCase):
    """Probabilistic Sequential QMixer — sequence + variational."""

    def setUp(self):
        cfg = {
            "type": "ProbSeqQMixer",
            "model": {
                "model_type": "QMixModel",
                "state_shape": 8,
                "input_dim": N,
                "qmix_hidden_dim": 16,
            },
            "feature_extractor": {
                "model_type": "Custom",
                "layers": [
                    {"type": "Linear", "in_features": STATE_FLAT, "out_features": 32},
                ],
            },
            "seq_model": {
                "model_type": "Custom",
                "layers": [
                    {"type": "Linear", "in_features": 32, "out_features": 16},
                    {"type": "ReLU"},
                ],
            },
            "deterministic_eval": False,
        }
        self.critic = CriticConfig(**cfg).get_critic()
        self.assertIsInstance(self.critic, ProbSeqQMixer)

    def test_forward_shape(self):
        out = self.critic(torch.randn(B, N), _mpe_state(), _mpe_alive(), _pad())
        self.assertEqual(out["q_tot"].shape, (B,))
        self.assertIn("mu", out)
        self.assertIn("std", out)

    def test_backward(self):
        q = torch.randn(B, N, requires_grad=True)
        out = self.critic(q, _mpe_state(), _mpe_alive(), _pad())
        out["q_tot"].sum().backward()
        self.assertIsNotNone(q.grad)


class TestGroupConsensusMixer(unittest.TestCase):
    """GroupConsensusMixer — consensus-aware state mixing."""

    def setUp(self):
        cfg = {
            "type": "GroupConsensusMixer",
            "model": {
                "model_type": "QMixModel",
                "state_shape": 16,
                "input_dim": N,
                "qmix_hidden_dim": 16,
            },
            "feature_extractor": {"model_type": "Identity"},
            "consensus_processor": {
                "model_type": "Custom",
                "layers": [
                    {"type": "Linear", "in_features": N * 8, "out_features": 16},
                    {"type": "ReLU"},
                ],
            },
            "num_agents": N,
            "group_latent_dim": 8,
        }
        self.critic = CriticConfig(**cfg).get_critic()
        self.assertIsInstance(self.critic, GroupConsensusMixer)

    def test_forward_shape(self):
        out = self.critic(torch.randn(B, N), _mpe_state(), _mpe_alive(), _pad())
        self.assertEqual(out["q_tot"].shape, (B,))

    def test_forward_with_consensus(self):
        mu = torch.randn(B, N, 8)
        log_var = torch.randn(B, N, 8)
        gids = torch.randint(0, 2, (B, N), dtype=torch.long)
        out = self.critic(
            torch.randn(B, N), _mpe_state(), _mpe_alive(), _pad(),
            group_mu=mu, group_log_var=log_var, group_indices=gids,
        )
        self.assertEqual(out["q_tot"].shape, (B,))

    def test_backward(self):
        q = torch.randn(B, N, requires_grad=True)
        out = self.critic(q, _mpe_state(), _mpe_alive(), _pad())
        out["q_tot"].sum().backward()
        self.assertIsNotNone(q.grad)


if __name__ == "__main__":
    unittest.main()
