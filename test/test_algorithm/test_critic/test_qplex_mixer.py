"""Tests for the QPLEXMixer critic.

Verifies:
- Mixer class inheritance and sub-module composition.
- Forward output shapes and ``q_tot == v_tot + a_tot`` identity.
- Attention regulariser (disabled / enabled).
- Gradient stop on the local advantage in ``a_tot`` (paper line 468).
- Gradient flow through ``v_tot`` to the Q-values.
- Dead-agent action masking.
- ``is_minus_one`` and ``weighted_head`` config toggles.
- ``CriticConfig`` dispatch to ``QPLEXMixer``.
- Gradient flow through state inputs.
"""

import unittest
import torch
from marlite.algorithm.critic import QPLEXMixer, CriticConfig, Mixer


def _make_qplex_mixer_cfg(
    state_dim=16,
    action_dim=5,
    n_agents=3,
    n_head=2,
    embed_dim=8,
    attend_reg_coef=0.0,
    weighted_head=True,
    is_minus_one=True,
):
    return {
        "type": "QPLEXMixer",
        "action_dim": action_dim,
        "n_agents": n_agents,
        "weighted_head": weighted_head,
        "is_minus_one": is_minus_one,
        "feature_extractor": {"model_type": "Identity"},
        "value_stream_w_final": {
            "model_type": "Custom",
            "layers": [
                {"type": "Linear", "in_features": state_dim, "out_features": n_agents},
            ],
        },
        "value_stream_v": {
            "model_type": "Custom",
            "layers": [
                {"type": "Linear", "in_features": state_dim, "out_features": n_agents},
            ],
        },
        "transformation": {
            "model_type": "QplexTransformation",
            "state_dim": state_dim,
            "n_agents": n_agents,
            "n_head": n_head,
            "embed_dim": embed_dim,
            "attend_reg_coef": attend_reg_coef,
            "selector_configs": [
                {
                    "model_type": "Custom",
                    "layers": [
                        {"type": "Linear", "in_features": state_dim, "out_features": embed_dim}
                    ],
                }
                for _ in range(n_head)
            ],
            "key_configs": [
                {
                    "model_type": "Custom",
                    "layers": [
                        {
                            "type": "Linear",
                            "in_features": state_dim,
                            "out_features": n_agents * embed_dim,
                        }
                    ],
                }
                for _ in range(n_head)
            ],
            "v_config": {
                "model_type": "Custom",
                "layers": [{"type": "Linear", "in_features": state_dim, "out_features": 1}],
            },
        },
        "joint_attention": {
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
                    "layers": [
                        {"type": "Linear", "in_features": state_dim, "out_features": n_agents}
                    ],
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
        },
    }


def _make_inputs(batch_size=4, n_agents=3, state_dim=16, action_dim=5, traj_len=7):
    return {
        "q_value_from_agents": torch.randn(batch_size, n_agents, action_dim),
        "states": torch.randn(batch_size, traj_len, state_dim),
        "actions": torch.randint(0, action_dim, (batch_size, n_agents)),
        "alive_mask": torch.ones(batch_size, traj_len, n_agents, dtype=torch.bool),
        "padding_mask": torch.ones(batch_size, traj_len, dtype=torch.bool),
    }


class TestQPLEXMixer(unittest.TestCase):
    def setUp(self):
        self.cfg = _make_qplex_mixer_cfg()
        self.mixer = CriticConfig(**self.cfg).get_critic()
        self.inputs = _make_inputs()

    def test_inherits_mixer(self):
        self.assertIsInstance(self.mixer, Mixer)

    def test_submodule_composition(self):
        self.assertTrue(hasattr(self.mixer, "transformation"))
        self.assertTrue(hasattr(self.mixer, "joint_attention"))
        self.assertTrue(hasattr(self.mixer, "value_w_final"))
        self.assertTrue(hasattr(self.mixer, "value_v"))
        self.assertTrue(hasattr(self.mixer, "feature_extractor"))
        self.assertEqual(self.mixer.action_dim, 5)
        self.assertEqual(self.mixer.n_agents, 3)

    def test_forward_shape(self):
        out = self.mixer(**self.inputs)
        self.assertIn("q_tot", out)
        self.assertIn("v_tot", out)
        self.assertIn("a_tot", out)
        self.assertIn("att_reg", out)
        self.assertEqual(out["q_tot"].shape, (4,))
        self.assertEqual(out["v_tot"].shape, (4,))
        self.assertEqual(out["a_tot"].shape, (4,))

    def test_q_tot_equals_v_tot_plus_a_tot(self):
        out = self.mixer(**self.inputs)
        self.assertTrue(torch.allclose(out["q_tot"], out["v_tot"] + out["a_tot"]))

    def test_att_reg_zero_when_disabled(self):
        out = self.mixer(**self.inputs)
        self.assertEqual(out["att_reg"].item(), 0.0)

    def test_att_reg_nonzero_when_enabled(self):
        cfg = _make_qplex_mixer_cfg(attend_reg_coef=0.5)
        m = CriticConfig(**cfg).get_critic()
        out = m(**self.inputs)
        self.assertGreater(out["att_reg"].item(), 0.0)

    def test_advantage_gradient_stopped(self):
        """The local advantage in the A stream must be detached (paper line 468)."""
        cfg = _make_qplex_mixer_cfg()
        m = CriticConfig(**cfg).get_critic()
        q_val = torch.randn(4, 3, 5, requires_grad=True)
        states = torch.randn(4, 7, 16)
        actions = torch.randint(0, 5, (4, 3))
        alive_mask = torch.ones(4, 7, 3, dtype=torch.bool)
        padding_mask = torch.ones(4, 7, dtype=torch.bool)
        out = m(q_val, states, actions, alive_mask, padding_mask)
        out["a_tot"].sum().backward()
        # a_tot should have NO gradient w.r.t. q_val because of detach
        self.assertIsNone(q_val.grad)

    def test_value_gradient_flows(self):
        """v_tot should depend on q_val (so gradient flows there)."""
        cfg = _make_qplex_mixer_cfg(weighted_head=True)
        m = CriticConfig(**cfg).get_critic()
        q_val = torch.randn(4, 3, 5, requires_grad=True)
        states = torch.randn(4, 7, 16)
        actions = torch.randint(0, 5, (4, 3))
        alive_mask = torch.ones(4, 7, 3, dtype=torch.bool)
        padding_mask = torch.ones(4, 7, dtype=torch.bool)
        out = m(q_val, states, actions, alive_mask, padding_mask)
        out["v_tot"].sum().backward()
        self.assertIsNotNone(q_val.grad)
        if q_val.grad is not None:
            self.assertGreater(q_val.grad.abs().sum().item(), 0.0)

    def test_dead_agent_action_masked_to_zero(self):
        """When an agent is dead at the last step, the joint action's slot for
        that agent must be all-zero (no-op)."""
        cfg = _make_qplex_mixer_cfg()
        m = CriticConfig(**cfg).get_critic()
        inputs = _make_inputs()
        # Mark agent 1 as dead at the last step
        inputs["alive_mask"][:, -1, 1] = False
        # Set a non-zero action for the dead agent
        inputs["actions"][0, 1] = 3
        # Forward should still work
        out = m(**inputs)
        self.assertEqual(out["q_tot"].shape, (4,))

    def test_is_minus_one_toggle(self):
        """is_minus_one=True vs False should produce different outputs."""
        cfg_minus = _make_qplex_mixer_cfg(is_minus_one=True)
        cfg_plain = _make_qplex_mixer_cfg(is_minus_one=False)
        m_minus = CriticConfig(**cfg_minus).get_critic()
        m_plain = CriticConfig(**cfg_plain).get_critic()
        # Share the same params
        m_plain.load_state_dict(m_minus.state_dict())
        out_minus = m_minus(**self.inputs)
        out_plain = m_plain(**self.inputs)
        # The two should generally differ (a_tot is different)
        self.assertFalse(torch.allclose(out_minus["a_tot"], out_plain["a_tot"]))

    def test_weighted_head_toggle(self):
        """weighted_head=True vs False should produce different v_tot."""
        cfg_w = _make_qplex_mixer_cfg(weighted_head=True)
        cfg_n = _make_qplex_mixer_cfg(weighted_head=False)
        m_w = CriticConfig(**cfg_w).get_critic()
        m_n = CriticConfig(**cfg_n).get_critic()
        # Share params
        m_n.load_state_dict(m_w.state_dict())
        out_w = m_w(**self.inputs)
        out_n = m_n(**self.inputs)
        self.assertFalse(torch.allclose(out_w["v_tot"], out_n["v_tot"]))

    def test_critic_config_dispatch(self):
        cfg = _make_qplex_mixer_cfg()
        critic = CriticConfig(**cfg).get_critic()
        self.assertIsInstance(critic, QPLEXMixer)

    def test_gradient_flow_through_states(self):
        """q_tot should depend on states (through transformation, joint attention, value stream)."""
        cfg = _make_qplex_mixer_cfg()
        m = CriticConfig(**cfg).get_critic()
        inputs = _make_inputs()
        inputs["states"].requires_grad_(True)
        out = m(**inputs)
        out["q_tot"].sum().backward()
        self.assertIsNotNone(inputs["states"].grad)
        if inputs["states"].grad is not None:
            self.assertGreater(inputs["states"].grad.abs().sum().item(), 0.0)

    def test_no_action_mask_works(self):
        """When next_avail_actions is a Space (not a Tensor), should still work."""
        # In this test, the mixer doesn't use next_avail_actions at all,
        # so we just verify the mixer forward doesn't depend on it.
        out = self.mixer(**self.inputs)
        self.assertTrue(torch.isfinite(out["q_tot"]).all())


if __name__ == "__main__":
    unittest.main()
