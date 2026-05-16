"""Tests for MAPPOCritic and SeqMAPPOCritic."""

import torch
import unittest

from marlite.algorithm.model import ModelConfig
from marlite.algorithm.critic import MAPPOCritic, SeqMAPPOCritic
from marlite.algorithm.critic.critic_config import CriticConfig, registered_critic_creators


class TestMAPPOCriticRegistration(unittest.TestCase):
    """Verify both critic types are registered in the factory."""

    def test_mappo_critic_registered(self):
        self.assertIn("MAPPOCritic", registered_critic_creators)

    def test_seq_mappo_critic_registered(self):
        self.assertIn("SeqMAPPOCritic", registered_critic_creators)


class TestMAPPOCriticConfig(unittest.TestCase):
    """Build MAPPOCritic from a config dict (no YAML)."""

    def setUp(self):
        self.cfg = {
            "type": "MAPPOCritic",
            "model": {
                "model_type": "Custom",
                "layers": [
                    {"type": "Linear", "in_features": 54, "out_features": 1},
                ],
            },
            "feature_extractor": {"model_type": "Identity"},
        }

    def test_create_from_dict(self):
        cc = CriticConfig(**self.cfg)
        critic = cc.get_critic()
        self.assertIsInstance(critic, MAPPOCritic)


class TestSeqMAPPOCriticConfig(unittest.TestCase):
    """Build SeqMAPPOCritic from a config dict."""

    def setUp(self):
        self.cfg = {
            "type": "SeqMAPPOCritic",
            "model": {
                "model_type": "Custom",
                "layers": [
                    {"type": "Linear", "in_features": 32, "out_features": 1},
                ],
            },
            "feature_extractor": {"model_type": "Identity"},
            "seq_model": {
                "model_type": "Custom",
                "layers": [
                    {"type": "Linear", "in_features": 54, "out_features": 32},
                    {"type": "ReLU"},
                ],
            },
        }

    def test_create_from_dict(self):
        cc = CriticConfig(**self.cfg)
        critic = cc.get_critic()
        self.assertIsInstance(critic, SeqMAPPOCritic)


class TestMAPPOCriticForward(unittest.TestCase):
    """Test forward() with both 3D and 4D state tensors."""

    def setUp(self):
        self.critic = MAPPOCritic(
            base_model_config=ModelConfig(
                model_type="Custom",
                layers=[{"type": "Linear", "in_features": 54, "out_features": 1}],
            ),
            feature_extractor_config=ModelConfig(model_type="Identity"),
        )

    def test_output_shape_3d(self):
        """3D states: (B, T, F)."""
        B, T, F = 4, 5, 54
        states = torch.randn(B, T, F)
        alive = torch.ones(B, T, dtype=torch.bool)
        pad = torch.zeros(B, T, dtype=torch.bool)
        out = self.critic(states, alive, pad)
        self.assertEqual(out["v"].shape, (B, 1))

    def test_output_shape_4d(self):
        """4D states: (B, T, N, F) — per-agent."""
        B, T, N, F = 4, 5, 3, 18
        states = torch.randn(B, T, N, F)
        alive = torch.ones(B, T, N, dtype=torch.bool)
        pad = torch.zeros(B, T, dtype=torch.bool)
        out = self.critic(states, alive, pad)
        self.assertEqual(out["v"].shape, (B, 1))

    def test_requires_grad(self):
        """Value computed by the critic is differentiable."""
        B, T, F = 4, 5, 54
        states = torch.randn(B, T, F, requires_grad=True)
        alive = torch.ones(B, T, dtype=torch.bool)
        pad = torch.zeros(B, T, dtype=torch.bool)
        out = self.critic(states, alive, pad)
        loss = out["v"].sum()
        loss.backward()
        self.assertIsNotNone(states.grad)


class TestSeqMAPPOCriticForward(unittest.TestCase):
    """Test forward() with both 3D and 4D state tensors."""

    def setUp(self):
        self.critic = SeqMAPPOCritic(
            base_model_config=ModelConfig(
                model_type="Custom",
                layers=[{"type": "Linear", "in_features": 32, "out_features": 1}],
            ),
            feature_extractor_config=ModelConfig(model_type="Identity"),
            seq_model_config=ModelConfig(
                model_type="Custom",
                layers=[
                    {"type": "Linear", "in_features": 54, "out_features": 32},
                    {"type": "ReLU"},
                ],
            ),
        )

    def test_output_shape_3d(self):
        B, T, F = 4, 5, 54
        states = torch.randn(B, T, F)
        alive = torch.ones(B, T, dtype=torch.bool)
        pad = torch.zeros(B, T, dtype=torch.bool)
        out = self.critic(states, alive, pad)
        self.assertEqual(out["v"].shape, (B, 1))

    def test_output_shape_4d(self):
        B, T, N, F = 4, 5, 3, 18
        states = torch.randn(B, T, N, F)
        alive = torch.ones(B, T, N, dtype=torch.bool)
        pad = torch.zeros(B, T, dtype=torch.bool)
        out = self.critic(states, alive, pad)
        self.assertEqual(out["v"].shape, (B, 1))

    def test_requires_grad(self):
        B, T, F = 4, 5, 54
        states = torch.randn(B, T, F, requires_grad=True)
        alive = torch.ones(B, T, dtype=torch.bool)
        pad = torch.zeros(B, T, dtype=torch.bool)
        out = self.critic(states, alive, pad)
        loss = out["v"].sum()
        loss.backward()
        self.assertIsNotNone(states.grad)


if __name__ == "__main__":
    unittest.main()
