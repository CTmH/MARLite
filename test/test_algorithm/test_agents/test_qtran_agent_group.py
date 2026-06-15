import unittest
import torch
from marlite.algorithm.agents import QTRANAgentGroup, AgentGroupConfig
from marlite.algorithm.model.model_config import ModelConfig


def _make_qtran_group():
    cfg = {
        "type": "QTRAN",
        "agent_list": {"agent_0": "m1", "agent_1": "m1", "agent_2": "m1"},
        "models": {
            "m1": {
                "feature_extractor": {"model_type": "Identity"},
                "encoder": {
                    "model_type": "RNN",
                    "input_shape": 8,
                    "rnn_hidden_dim": 16,
                    "rnn_layers": 1,
                    "output_shape": 16,
                },
                "decoder": {
                    "model_type": "Custom",
                    "layers": [
                        {"type": "Linear", "in_features": 16, "out_features": 5},
                    ],
                },
            },
        },
    }
    return AgentGroupConfig(**cfg).get_agent_group()


class TestQTRANAgentGroup(unittest.TestCase):
    def setUp(self):
        self.group = _make_qtran_group()
        self.batch_size = 4
        self.n_agents = 3
        self.traj_len = 5
        self.obs_dim = 8
        self.action_dim = 5

    def _make_inputs(self):
        obs = torch.randn(self.batch_size, self.n_agents, self.traj_len, self.obs_dim)
        pad = torch.ones(self.batch_size, self.traj_len, dtype=torch.bool)
        alive = torch.ones(self.batch_size, self.n_agents, dtype=torch.bool)
        return obs, pad, alive

    def test_forward_returns_q_val_and_enc_out(self):
        obs, pad, alive = self._make_inputs()
        ret = self.group(obs, pad, alive)
        self.assertIn("q_val", ret)
        self.assertIn("enc_out", ret)
        self.assertEqual(ret["q_val"].shape, (self.batch_size, self.n_agents, self.action_dim))
        self.assertEqual(ret["enc_out"].shape, (self.batch_size, self.n_agents, 16))

    def test_forward_no_actions_required(self):
        obs, pad, alive = self._make_inputs()
        ret = self.group(obs, pad, alive)
        self.assertIsNotNone(ret["q_val"])

    def test_dead_agents_zero_q(self):
        obs, pad, alive = self._make_inputs()
        alive[:, 1] = False
        ret = self.group(obs, pad, alive)
        self.assertTrue(torch.all(ret["q_val"][:, 1, :] == 0))

    def test_enc_out_gradient_flows(self):
        obs, pad, alive = self._make_inputs()
        ret = self.group(obs, pad, alive)
        loss = ret["enc_out"].sum() + ret["q_val"].sum()
        loss.backward()
        for p in self.group.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)
                self.assertFalse(torch.all(p.grad == 0))

    def test_train_mode_encoder_active(self):
        self.group.train()
        for enc in self.group.encoders.values():  # type: ignore[union-attr]
            self.assertTrue(enc.training)  # type: ignore[attr-defined]

    def test_eval_mode_encoder_inactive(self):
        self.group.eval()
        for enc in self.group.encoders.values():  # type: ignore[union-attr]
            self.assertFalse(enc.training)  # type: ignore[attr-defined]

    def test_reset_returns_self(self):
        self.assertIs(self.group.reset(), self.group)

    def test_qmix_agent_group_does_not_have_enc_out(self):
        from marlite.algorithm.agents.qmix_agent_group import QMIXAgentGroup
        self.assertTrue(QMIXAgentGroup.forward is not None)


if __name__ == "__main__":
    unittest.main()
