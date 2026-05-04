import unittest
import numpy as np

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
        states = np.random.randn(bs, 10, 10, 5)
        zone_indices = builder(states)

        self.assertEqual(zone_indices.shape, (bs, len(self.group_ids)))
        self.assertEqual(zone_indices.dtype, np.int16)

        for b in range(bs):
            np.testing.assert_array_equal(
                zone_indices[b], np.array(self.group_ids, dtype=np.int16)
            )

    def test_reset(self):
        builder_config = GroupBuilderConfig(**self.config)
        builder = builder_config.get_group_builder()
        builder.reset()
        states = np.random.randn(2, 10, 10, 5)
        zone_indices = builder(states)
        self.assertEqual(zone_indices.shape, (2, len(self.group_ids)))


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
