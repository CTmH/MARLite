"""Validate the 50v50 QPLEX example without starting SC2 or rollout workers."""

from pathlib import Path
import unittest

import torch
import yaml

from marlite.trainer import TrainerConfig
from marlite.trainer.qplex_trainer import QPLEXTrainer
from marlite.algorithm.critic import SeqQPLEXMixer


class TestFlat64QPLEXConfig(unittest.TestCase):
    def test_config_and_masked_forward_backward(self):
        path = Path(__file__).resolve().parents[2] / "examples/smac_flat64_50v50_qplex_td1_long.yaml"
        config = yaml.safe_load(path.read_text())
        parsed = TrainerConfig(config)
        self.assertIs(parsed.trainer_class, QPLEXTrainer)
        self.assertEqual(parsed.train_args["epochs"], 1000)
        self.assertGreater(parsed.train_args["rollback_interval"], 1000)
        self.assertEqual(parsed.trainer_kwargs["n_steps"], 1)
        self.assertEqual(config["rollout"]["n_workers"], 4)
        self.assertEqual(config["rollout"]["traj_len"], 16)
        self.assertEqual(config["replay_buffer"]["traj_len"], 16)
        epsilon = parsed.trainer_kwargs["epsilon_scheduler"]
        self.assertAlmostEqual(epsilon.get_value(0), 1.0)
        self.assertAlmostEqual(epsilon.get_value(100), 0.05)
        self.assertAlmostEqual(epsilon.get_value(999), 0.05)

        agent = parsed.trainer_kwargs["agent_group_config"].get_agent_group().to("cpu")
        mixer = parsed.trainer_kwargs["critic_config"].get_critic().to("cpu")
        self.assertIsInstance(mixer, SeqQPLEXMixer)
        observations = torch.randn(2, 50, 16, 852)
        states = torch.randn(2, 16, 50, 1151)
        alive = torch.ones(2, 16, 50, dtype=torch.bool)
        alive[0, :, -5:] = False  # Partially dead team.
        alive[1] = False  # Terminal, all-dead team.
        padding = torch.zeros(2, 16, dtype=torch.bool)
        padding[0, :8] = True  # Short history at the start of an episode.
        q_values = agent(observations, padding[:, None].expand(-1, 50, -1), alive[:, -1])["q_val"]
        self.assertEqual(tuple(q_values.shape), (2, 50, 56))
        available = torch.zeros_like(q_values, dtype=torch.bool)
        available[..., :6] = alive[:, -1, :, None]
        actions = torch.zeros(2, 50, dtype=torch.long)
        output = mixer(q_values, states, actions, alive, padding, avail_actions=available)
        self.assertTrue(torch.isfinite(output["q_tot"]).all())
        self.assertEqual(output["q_tot"][1].item(), 0.0)
        loss = (output["q_tot"][0] - 1.0).square() + output["att_reg"]
        loss.backward()
        self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad).item()
                            for p in mixer.seq_model.parameters()))
        for module in (agent, mixer):
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertTrue(any(torch.count_nonzero(g).item() for g in gradients))
        # Repeated YAML aliases must create independent attention-head parameters.
        self.assertIsNot(mixer.joint_attention.keys[0], mixer.joint_attention.keys[1])
        self.assertIsNot(next(mixer.value_w_final.parameters()), next(mixer.value_v.parameters()))


if __name__ == "__main__":
    unittest.main()
