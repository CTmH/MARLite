import unittest
import torch
from marlite.algorithm.model import GeneralHyperNetwork, QMixHyperNetwork, HyperNetwork, ModelConfig

class TestHypernetCompatibility(unittest.TestCase):
    def test_signed_and_monotonic_weights(self):
        signed = GeneralHyperNetwork(1, [1, 1])
        positive = QMixHyperNetwork(1, [1, 1])
        with torch.no_grad():
            for p in signed.parameters(): p.zero_()
            signed.hyper_ws[0].bias.fill_(-2)
        positive.load_state_dict(signed.state_dict(), strict=True)
        x, z = torch.ones(2, 1), torch.zeros(2, 1)
        torch.testing.assert_close(signed(x, z), -2*x)
        torch.testing.assert_close(positive(x, z), 2*x)
        self.assertIs(HyperNetwork, QMixHyperNetwork)

    def test_legacy_config_and_checkpoint(self):
        old = ModelConfig(model_type="HyperNetwork", cond_dim=3, layer_dims=[4, 5, 2]).get_model()
        new = ModelConfig(model_type="QMixHyperNetwork", cond_dim=3, layer_dims=[4, 5, 2]).get_model()
        new.load_state_dict(old.state_dict(), strict=True)
        x, z = torch.randn(6, 4), torch.randn(6, 3)
        torch.testing.assert_close(old(x, z), new(x, z))
        signed = ModelConfig(model_type="GeneralHyperNetwork", cond_dim=3, layer_dims=[4, 2]).get_model()
        signed(x.requires_grad_(), z).sum().backward()
        self.assertIsNotNone(x.grad)
