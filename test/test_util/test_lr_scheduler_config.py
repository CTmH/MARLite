import unittest
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim import Adam
from torch.nn import Linear
from marlite.util.lr_scheduler_config import LRSchedulerConfig, registered_lr_scheduler

class TestOptimizerConfig(unittest.TestCase):
    def setUp(self):
        self.model = Linear(32, 8)
        self.optimizer = Adam(self.model.parameters(), lr=0.01)

    def test_step_lr_scheduler(self):
        """Test that StepLR scheduler is created correctly"""
        config = LRSchedulerConfig(type="StepLR", step_size=10, gamma=0.1)
        scheduler = config.get_lr_scheduler(self.optimizer)

        self.assertIsInstance(scheduler, lr_scheduler.StepLR)
        self.assertEqual(scheduler.step_size, 10)
        self.assertEqual(scheduler.gamma, 0.1)

    def test_one_cycle_lr_scheduler(self):
        """Test that OneCycleLR scheduler is created correctly"""
        config = LRSchedulerConfig(
            type="OneCycleLR",
            max_lr=0.1,
            total_steps=1000,
            pct_start=0.3
        )
        scheduler = config.get_lr_scheduler(self.optimizer)

        self.assertIsInstance(scheduler, lr_scheduler.OneCycleLR)

    def test_reduce_lr_on_plateau_scheduler(self):
        """Test that ReduceLROnPlateau scheduler is created correctly"""
        config = LRSchedulerConfig(
            type="ReduceLROnPlateau",
            mode="min",
            factor=0.5,
            patience=5
        )
        scheduler = config.get_lr_scheduler(self.optimizer)

        self.assertIsInstance(scheduler, lr_scheduler.ReduceLROnPlateau)
        self.assertEqual(scheduler.mode, "min")
        self.assertEqual(scheduler.factor, 0.5)
        self.assertEqual(scheduler.patience, 5)

    def test_registered_lr_scheduler_contains_all_types(self):
        """Test that all supported schedulers are in the registry"""
        expected_schedulers = {
            "StepLR": lr_scheduler.StepLR,
            "OneCycleLR": lr_scheduler.OneCycleLR,
            "ReduceLROnPlateau": lr_scheduler.ReduceLROnPlateau,
        }

        self.assertEqual(len(registered_lr_scheduler), len(expected_schedulers))
        for name, cls in expected_schedulers.items():
            self.assertIn(name, registered_lr_scheduler)
            self.assertEqual(registered_lr_scheduler[name], cls)

if __name__ == '__main__':
    unittest.main()