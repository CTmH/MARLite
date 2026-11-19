import unittest
import numpy as np
import torch

from marlite.configuration.catalog import default_registry


class TestFixedGroupBuilder(unittest.TestCase):
    def setUp(self):
        self.group_ids = [0, 0, 1, 1, 2, 2]
        self.config = {
            "type": "FixedGroupBuilder",
            "group_ids": self.group_ids,
        }

    def test_forward(self):
        builder = default_registry().build(
            {"type": self.config["type"], "params": {"group_ids": self.group_ids}}
        )

        bs = 5
        states = torch.from_numpy(np.random.randn(bs, 10, 10, 5)).float()
        group_indices = builder(states)

        self.assertEqual(group_indices.shape, (bs, len(self.group_ids)))
        self.assertEqual(group_indices.dtype, torch.int16)

        for b in range(bs):
            expected = torch.tensor(self.group_ids, dtype=torch.int16)
            self.assertTrue(torch.equal(group_indices[b], expected))

    def test_reset(self):
        builder = default_registry().build(
            {"type": self.config["type"], "params": {"group_ids": self.group_ids}}
        )
        builder.reset()
        states = torch.from_numpy(np.random.randn(2, 10, 10, 5)).float()
        group_indices = builder(states)
        self.assertEqual(group_indices.shape, (2, len(self.group_ids)))


class TestGroupBuilderRegistry(unittest.TestCase):
    def test_fixed_group_builder_creation(self):
        config = {
            "type": "FixedGroupBuilder",
            "group_ids": [0, 0, 1, 1],
        }
        builder = default_registry().build(
            {"type": config["type"], "params": {"group_ids": config["group_ids"]}}
        )
        self.assertIsNotNone(builder)

    def test_invalid_type(self):
        config = {
            "type": "InvalidType",
            "group_ids": [0, 0, 1, 1],
        }
        with self.assertRaises(ValueError):
            default_registry().build(config)


if __name__ == "__main__":
    unittest.main()
