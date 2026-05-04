import os
import unittest
import yaml
from marlite.trainer.trainer_config import TrainerConfig, REGISTERED_TRAINERS


class TestGroupConsensusTrainer(unittest.TestCase):
    def setUp(self):
        self.config_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "config", "group_consensus_default.yaml"
        )

    def test_trainer_creation(self):
        conf = TrainerConfig(self.config_path)
        trainer = conf.get_trainer()
        self.assertIsNotNone(trainer)


class TestGroupConsensusTrainerConfig(unittest.TestCase):
    def test_group_consensus_trainer_registration(self):
        self.assertIn("GroupConsensus", REGISTERED_TRAINERS)


if __name__ == "__main__":
    unittest.main()
