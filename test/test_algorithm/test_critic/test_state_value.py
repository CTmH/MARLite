import unittest
import torch
from marlite.algorithm.critic import StateValue, SeqStateValue, StateValueConfig
from marlite.algorithm.model.model_config import ModelConfig

B = 4
T = 5
STATE_DIM = 12
SEQ_HIDDEN = 16


class TestStateValue(unittest.TestCase):
    def setUp(self):
        self.base_model_config = ModelConfig(
            model_type="Custom",
            layers=[
                {"type": "Linear", "in_features": STATE_DIM, "out_features": 8},
                {"type": "ReLU"},
                {"type": "Linear", "in_features": 8, "out_features": 1},
            ],
        )
        self.fe_config = ModelConfig(
            model_type="Identity",
        )

    def test_forward_shape(self):
        net = StateValue(self.base_model_config, self.fe_config)
        states = torch.randn(B, T, STATE_DIM)
        alive = torch.ones(B, T)
        pad = torch.ones(B, T, dtype=torch.bool)
        out = net(states, alive, pad)
        self.assertIn("v", out)
        self.assertEqual(out["v"].shape, (B, 1))

    def test_backward(self):
        net = StateValue(self.base_model_config, self.fe_config)
        states = torch.randn(B, T, STATE_DIM, requires_grad=True)
        alive = torch.ones(B, T)
        pad = torch.ones(B, T, dtype=torch.bool)
        out = net(states, alive, pad)
        out["v"].sum().backward()
        for p in net.parameters():
            self.assertIsNotNone(p.grad)
            self.assertFalse(torch.all(p.grad == 0))

    def test_state_value_config(self):
        cfg = {
            "type": "StateValue",
            "base_model": {
                "model_type": "Custom",
                "layers": [
                    {"type": "Linear", "in_features": STATE_DIM, "out_features": 1},
                ],
            },
            "feature_extractor": {"model_type": "Identity"},
        }
        net = StateValueConfig(**cfg).get_v_net()
        self.assertIsInstance(net, StateValue)


class TestSeqStateValue(unittest.TestCase):
    def setUp(self):
        self.base_model_config = ModelConfig(
            model_type="Custom",
            layers=[
                {"type": "Linear", "in_features": SEQ_HIDDEN, "out_features": 1},
            ],
        )
        self.fe_config = ModelConfig(model_type="Identity")
        self.seq_model_config = ModelConfig(
            model_type="RNN",
            input_shape=STATE_DIM,
            rnn_hidden_dim=SEQ_HIDDEN,
            rnn_layers=1,
            output_shape=SEQ_HIDDEN,
        )

    def test_forward_shape(self):
        net = SeqStateValue(self.base_model_config, self.fe_config, self.seq_model_config)
        states = torch.randn(B, T, STATE_DIM)
        alive = torch.ones(B, T)
        pad = torch.ones(B, T, dtype=torch.bool)
        out = net(states, alive, pad)
        self.assertEqual(out["v"].shape, (B, 1))

    def test_backward(self):
        net = SeqStateValue(self.base_model_config, self.fe_config, self.seq_model_config)
        states = torch.randn(B, T, STATE_DIM, requires_grad=True)
        alive = torch.ones(B, T)
        pad = torch.ones(B, T, dtype=torch.bool)
        out = net(states, alive, pad)
        out["v"].sum().backward()
        for p in net.parameters():
            self.assertIsNotNone(p.grad)
            self.assertFalse(torch.all(p.grad == 0))

    def test_seq_state_value_config(self):
        cfg = {
            "type": "SeqStateValue",
            "base_model": {
                "model_type": "Custom",
                "layers": [
                    {"type": "Linear", "in_features": SEQ_HIDDEN, "out_features": 1},
                ],
            },
            "feature_extractor": {"model_type": "Identity"},
            "seq_model": {
                "model_type": "RNN",
                "input_shape": STATE_DIM,
                "rnn_hidden_dim": SEQ_HIDDEN,
                "rnn_layers": 1,
                "output_shape": SEQ_HIDDEN,
            },
        }
        net = StateValueConfig(**cfg).get_v_net()
        self.assertIsInstance(net, SeqStateValue)

    def test_unknown_type_raises(self):
        with self.assertRaises(ValueError):
            StateValueConfig(type="UnknownVNet", base_model={})


if __name__ == "__main__":
    unittest.main()
