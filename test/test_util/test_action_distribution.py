import unittest

import torch

from marlite.util.action_distribution import masked_categorical


class TestActionDistribution(unittest.TestCase):
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
