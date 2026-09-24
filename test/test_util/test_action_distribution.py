import unittest

import numpy as np
import torch

from marlite.util.action_distribution import masked_categorical
from marlite.util.trajectory_dataset import trajectory_collate_fn


class TestActionDistribution(unittest.TestCase):
    def test_all_mappo_collectors_match_training_distribution(self):
        from marlite.algorithm.agents.mappo_agent_group import MAPPOAgentGroup
        from marlite.algorithm.agents.g2anet_mappo_agent_group import G2ANetMAPPOAgentGroup
        from marlite.algorithm.agents.group_consensus_mappo_agent_group import GroupConsensusMAPPOAgentGroup

        class FixedPolicy:
            device = torch.device("cpu")
            agent_model_dict = {"a": "shared", "b": "shared"}

            def __call__(self, *args):
                return {"action_logits": torch.zeros(1, 2, 56),
                        "edge_indices": [torch.empty(2, 0, dtype=torch.long)],
                        "group_indices": torch.zeros(1, 2, dtype=torch.long)}

        mask = np.zeros(56, dtype=bool)
        mask[:6] = True
        for group in (MAPPOAgentGroup, G2ANetMAPPOAgentGroup, GroupConsensusMAPPOAgentGroup):
            with self.subTest(group=group.__name__):
                result = group.act(FixedPolicy(),
                    observations={a: np.zeros((2, 3)) for a in ("a", "b")},
                    state=np.zeros(3), avail_actions={"a": mask, "b": np.zeros(56, dtype=bool)},
                    traj_padding_mask=np.zeros(2, dtype=bool), alive_agents=["a"])
                action = torch.tensor(result["actions"]["a"])
                trained = masked_categorical(torch.zeros(56), torch.from_numpy(mask))
                self.assertAlmostEqual(trained.log_prob(action).item(), result["log_probs"]["a"])
                self.assertLess(action.item(), 6)
                self.assertEqual(result["all_log_probs"]["b"], 0)

    def test_masked_ratio_entropy_and_inactive_rows(self):
        logits = torch.zeros(3, 56, requires_grad=True)
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[0, :6] = True
        mask[1, 3] = True
        active = torch.tensor([True, True, False])
        actions = torch.tensor([2, 3, 0])
        old = masked_categorical(logits.detach(), mask, active).log_prob(actions)
        new = masked_categorical(logits, mask, active, actions)
        torch.testing.assert_close((new.log_prob(actions) - old).exp(), torch.ones(3))
        self.assertEqual(new.probs[1, 3].item(), 1)
        self.assertEqual(new.entropy()[1].item(), 0)
        changed = logits.detach().clone().masked_fill(~mask, 1000)
        torch.testing.assert_close(new.probs, masked_categorical(changed, mask, active).probs)
        loss = ((-new.log_prob(actions) - new.entropy()) * active).sum()
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(logits.grad[2].abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            masked_categorical(logits, mask)
        with self.assertRaises(ValueError):
            masked_categorical(logits, mask, active, torch.tensor([7, 3, 0]))
        torch.testing.assert_close(masked_categorical(logits).probs, logits.softmax(-1))

    def test_bool_masks_survive_collation(self):
        batch = trajectory_collate_fn([
            {"avail_actions": np.array([[[True, False]]])},
            {"avail_actions": np.array([[[False, True]]])},
        ])
        self.assertEqual(batch["avail_actions"].dtype, torch.bool)
        self.assertEqual(batch["avail_actions"].shape, (2, 1, 1, 2))
