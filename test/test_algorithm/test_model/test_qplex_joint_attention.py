"""Tests for the QplexJointAttention module.

Verifies:
- ModelConfig registration and instantiation.
- Output shape and positivity of lambda.
- Gradient flow through both state and joint-action inputs.
- Sub-network gradient receipt.
- Input dimension matches ``state_dim + N * A`` (no extraneous ``+1``).
- Single-head configuration.
- Input validation for misconfigured config lengths.
"""

import unittest
import torch
from marlite.algorithm.model import ModelConfig, QplexJointAttention


def _make_joint_attention(
    state_dim=16,
    action_dim=5,
    n_agents=3,
    n_head=2,
):
    cfg = {
        "model_type": "QplexJointAttention",
        "state_dim": state_dim,
        "action_dim": action_dim,
        "n_agents": n_agents,
        "n_head": n_head,
        "key_configs": [
            {
                "model_type": "Custom",
                "layers": [{"type": "Linear", "in_features": state_dim, "out_features": 1}],
            }
            for _ in range(n_head)
        ],
        "agent_extractor_configs": [
            {
                "model_type": "Custom",
                "layers": [{"type": "Linear", "in_features": state_dim, "out_features": n_agents}],
            }
            for _ in range(n_head)
        ],
        "action_extractor_configs": [
            {
                "model_type": "Custom",
                "layers": [
                    {
                        "type": "Linear",
                        "in_features": state_dim + n_agents * action_dim,
                        "out_features": n_agents,
                    }
                ],
            }
            for _ in range(n_head)
        ],
    }
    return ModelConfig(**cfg).get_model()


class TestQplexJointAttention(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.n_agents = 3
        self.state_dim = 16
        self.action_dim = 5
        self.jattn = _make_joint_attention(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            n_agents=self.n_agents,
            n_head=2,
        )

    def test_registered_as_model_type(self):
        cfg = ModelConfig(
            model_type="QplexJointAttention",
            state_dim=8,
            action_dim=3,
            n_agents=2,
            n_head=1,
            key_configs=[
                {
                    "model_type": "Custom",
                    "layers": [{"type": "Linear", "in_features": 8, "out_features": 1}],
                }
            ],
            agent_extractor_configs=[
                {
                    "model_type": "Custom",
                    "layers": [{"type": "Linear", "in_features": 8, "out_features": 2}],
                }
            ],
            action_extractor_configs=[
                {
                    "model_type": "Custom",
                    "layers": [{"type": "Linear", "in_features": 14, "out_features": 2}],
                }
            ],
        )
        model = cfg.get_model()
        self.assertIsInstance(model, QplexJointAttention)

    def test_output_shape(self):
        states = torch.randn(self.batch_size, self.state_dim)
        joint_actions = torch.randn(self.batch_size, self.n_agents * self.action_dim)
        lam = self.jattn(states, joint_actions)
        self.assertEqual(lam.shape, (self.batch_size, self.n_agents))

    def test_lambda_is_positive(self):
        states = torch.randn(self.batch_size, self.state_dim)
        joint_actions = torch.randn(self.batch_size, self.n_agents * self.action_dim)
        lam = self.jattn(states, joint_actions)
        self.assertTrue(torch.all(lam > 0))

    def test_gradient_flow(self):
        states = torch.randn(self.batch_size, self.state_dim)
        states.requires_grad_(True)
        joint_actions = torch.randn(
            self.batch_size, self.n_agents * self.action_dim, requires_grad=True
        )
        lam = self.jattn(states, joint_actions)
        lam.sum().backward()
        self.assertIsNotNone(states.grad)
        self.assertIsNotNone(joint_actions.grad)
        if states.grad is not None:
            self.assertGreater(states.grad.abs().sum().item(), 0.0)
        if joint_actions.grad is not None:
            self.assertGreater(joint_actions.grad.abs().sum().item(), 0.0)

    def test_all_subnetworks_get_gradients(self):
        states = torch.randn(self.batch_size, self.state_dim)
        joint_actions = torch.randn(self.batch_size, self.n_agents * self.action_dim)
        lam = self.jattn(states, joint_actions)
        lam.sum().backward()
        n_with_grad = sum(
            1 for p in self.jattn.parameters()
            if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
        )
        n_total = sum(1 for p in self.jattn.parameters() if p.requires_grad)
        self.assertEqual(n_with_grad, n_total)

    def test_input_dim_no_extra_plus_one(self):
        """Verify the action_extractor input is state_dim + N*A (not +1)."""
        state_dim = 16
        action_dim = 5
        n_agents = 3
        # Build a 2-layer model whose first layer's input dim we can inspect.
        layers = [{"type": "Linear", "in_features": state_dim + n_agents * action_dim, "out_features": 1}]
        cfg = {
            "model_type": "QplexJointAttention",
            "state_dim": state_dim,
            "action_dim": action_dim,
            "n_agents": n_agents,
            "n_head": 1,
            "key_configs": [
                {
                    "model_type": "Custom",
                    "layers": [{"type": "Linear", "in_features": state_dim, "out_features": 1}],
                }
            ],
            "agent_extractor_configs": [
                {
                    "model_type": "Custom",
                    "layers": [{"type": "Linear", "in_features": state_dim, "out_features": n_agents}],
                }
            ],
            "action_extractor_configs": [
                {"model_type": "Custom", "layers": layers},
            ],
        }
        model = ModelConfig(**cfg).get_model()
        # Inspect action_extractor's first layer weight
        first_linear = model.action_extractors[0].model[0]  # type: ignore[attr-defined]
        self.assertEqual(first_linear.in_features, state_dim + n_agents * action_dim)

    def test_single_head(self):
        m = _make_joint_attention(
            state_dim=self.state_dim, action_dim=self.action_dim, n_agents=self.n_agents, n_head=1
        )
        states = torch.randn(self.batch_size, self.state_dim)
        joint_actions = torch.randn(self.batch_size, self.n_agents * self.action_dim)
        lam = m(states, joint_actions)
        self.assertEqual(lam.shape, (self.batch_size, self.n_agents))

    def test_invalid_config_lengths(self):
        with self.assertRaises(ValueError):
            ModelConfig(
                model_type="QplexJointAttention",
                state_dim=8,
                action_dim=3,
                n_agents=2,
                n_head=3,
                key_configs=[
                    {
                        "model_type": "Custom",
                        "layers": [{"type": "Linear", "in_features": 8, "out_features": 1}],
                    }
                    for _ in range(3)
                ],
                agent_extractor_configs=[
                    {
                        "model_type": "Custom",
                        "layers": [{"type": "Linear", "in_features": 8, "out_features": 2}],
                    }
                ],  # wrong length
                action_extractor_configs=[
                    {
                        "model_type": "Custom",
                        "layers": [{"type": "Linear", "in_features": 14, "out_features": 2}],
                    }
                    for _ in range(3)
                ],
            ).get_model()


if __name__ == "__main__":
    unittest.main()
