import unittest
import torch
from marlite.algorithm.critic import Qtransform, CriticConfig
from marlite.algorithm.model.model_config import ModelConfig
from marlite.algorithm.critic.mixer import Mixer


def _make_qtransform(enc_out_dim=8, encoding_dim=16, action_dim=5):
    return Qtransform(
        phi_net_config=ModelConfig(
            model_type="Custom",
            layers=[
                {"type": "Linear", "in_features": enc_out_dim + action_dim, "out_features": encoding_dim},
                {"type": "ReLU"},
                {"type": "Linear", "in_features": encoding_dim, "out_features": encoding_dim},
            ],
        ),
        psi_net_config=ModelConfig(
            model_type="Custom",
            layers=[
                {"type": "Linear", "in_features": enc_out_dim, "out_features": encoding_dim},
                {"type": "ReLU"},
                {"type": "Linear", "in_features": encoding_dim, "out_features": encoding_dim},
            ],
        ),
        base_model_config=ModelConfig(
            model_type="Custom",
            layers=[
                {"type": "Linear", "in_features": encoding_dim, "out_features": encoding_dim},
                {"type": "ReLU"},
                {"type": "Linear", "in_features": encoding_dim, "out_features": action_dim},
            ],
        ),
        action_dim=action_dim,
    )


class TestQtransform(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.n_agents = 3
        self.enc_out_dim = 8
        self.encoding_dim = 16
        self.action_dim = 5
        self.net = _make_qtransform(
            enc_out_dim=self.enc_out_dim,
            encoding_dim=self.encoding_dim,
            action_dim=self.action_dim,
        )

    def test_inherits_mixer(self):
        self.assertIsInstance(self.net, Mixer)

    def test_has_submodules(self):
        self.assertTrue(hasattr(self.net, "phi_net"))
        self.assertTrue(hasattr(self.net, "psi_net"))
        self.assertTrue(hasattr(self.net, "base_model"))
        self.assertEqual(self.net.action_dim, self.action_dim)

    def test_forward_shape(self):
        enc_out = torch.randn(self.batch_size, self.n_agents, self.enc_out_dim)
        actions = torch.randint(0, self.action_dim, (self.batch_size, self.n_agents))
        out = self.net(enc_out, actions)
        self.assertIn("q_per_action", out)
        self.assertEqual(out["q_per_action"].shape, (self.batch_size, self.n_agents, self.action_dim))

    def test_backward_through_encoders(self):
        enc_out = torch.randn(self.batch_size, self.n_agents, self.enc_out_dim, requires_grad=True)
        actions = torch.randint(0, self.action_dim, (self.batch_size, self.n_agents))
        out = self.net(enc_out, actions)
        out["q_per_action"].sum().backward()
        for p in self.net.parameters():
            self.assertIsNotNone(p.grad)
            self.assertFalse(torch.all(p.grad == 0))
        self.assertIsNotNone(enc_out.grad)
        self.assertFalse(torch.all(enc_out.grad == 0))

    def test_critic_config_dispatch(self):
        cfg = {
            "type": "Qtransform",
            "action_dim": 4,
            "base_model": {
                "model_type": "Custom",
                "layers": [
                    {"type": "Linear", "in_features": 8, "out_features": 4},
                ],
            },
            "phi_net": {
                "model_type": "Custom",
                "layers": [
                    {"type": "Linear", "in_features": 4 + 4, "out_features": 8},
                    {"type": "ReLU"},
                ],
            },
            "psi_net": {
                "model_type": "Custom",
                "layers": [
                    {"type": "Linear", "in_features": 4, "out_features": 8},
                    {"type": "ReLU"},
                ],
            },
        }
        net = CriticConfig(**cfg).get_critic()
        self.assertIsInstance(net, Qtransform)
        self.assertEqual(net.action_dim, 4)


if __name__ == "__main__":
    unittest.main()
