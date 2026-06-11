import unittest
import numpy as np
import torch

from marlite.algorithm.group_builder import GroupBuilderConfig


class TestFixedGroupBuilder(unittest.TestCase):
    def setUp(self):
        self.group_ids = [0, 0, 1, 1, 2, 2]
        self.config = {
            "type": "Fixed",
            "group_ids": self.group_ids,
        }

    def test_forward(self):
        builder_config = GroupBuilderConfig(**self.config)
        builder = builder_config.get_group_builder()

        bs = 5
        states = torch.from_numpy(np.random.randn(bs, 10, 10, 5)).float()
        group_indices = builder(states)

        self.assertEqual(group_indices.shape, (bs, len(self.group_ids)))
        self.assertEqual(group_indices.dtype, torch.int16)

        for b in range(bs):
            expected = torch.tensor(self.group_ids, dtype=torch.int16)
            self.assertTrue(torch.equal(group_indices[b], expected))

    def test_reset(self):
        builder_config = GroupBuilderConfig(**self.config)
        builder = builder_config.get_group_builder()
        builder.reset()
        states = torch.from_numpy(np.random.randn(2, 10, 10, 5)).float()
        group_indices = builder(states)
        self.assertEqual(group_indices.shape, (2, len(self.group_ids)))


class TestGroupBuilderConfig(unittest.TestCase):
    def test_fixed_group_builder_creation(self):
        config = {
            "type": "Fixed",
            "group_ids": [0, 0, 1, 1],
        }
        builder_config = GroupBuilderConfig(**config)
        builder = builder_config.get_group_builder()
        self.assertIsNotNone(builder)

    def test_invalid_type(self):
        config = {
            "type": "InvalidType",
            "group_ids": [0, 0, 1, 1],
        }
        with self.assertRaises(ValueError):
            GroupBuilderConfig(**config)


if __name__ == "__main__":
    unittest.main()
