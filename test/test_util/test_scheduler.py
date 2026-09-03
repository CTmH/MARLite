import unittest

from marlite.util.scheduler import (
    FixedScheduler,
    LinearScheduler,
    LogarithmicScheduler,
)
from marlite.util.scheduler_config import SchedulerConfig, registered_schedulers


class TestSchedulers(unittest.TestCase):
    def test_linear_scheduler_has_exact_warmup_and_endpoints(self):
        scheduler = LinearScheduler(
            start_value=-2.0,
            end_value=6.0,
            ramp_start_step=40,
            ramp_steps=24,
        )

        self.assertEqual(scheduler.get_value(0), -2.0)
        self.assertEqual(scheduler.get_value(40), -2.0)
        self.assertEqual(scheduler.get_value(52), 2.0)
        self.assertEqual(scheduler.get_value(64), 6.0)
        self.assertEqual(scheduler.get_value(100), 6.0)

    def test_logarithmic_scheduler_delays_growth_and_reaches_end(self):
        scheduler = LogarithmicScheduler(
            start_value=0.0,
            end_value=1.0,
            ramp_start_step=40,
            ramp_steps=80,
            curve_rate=6.0,
        )

        self.assertEqual(scheduler.get_value(40), 0.0)
        self.assertLess(scheduler.get_value(60), 0.01)
        self.assertLess(scheduler.get_value(80), 0.05)
        self.assertGreater(scheduler.get_value(100), 0.2)
        self.assertEqual(scheduler.get_value(120), 1.0)

    def test_zero_logarithmic_rate_is_exactly_linear(self):
        logarithmic = LogarithmicScheduler(
            start_value=0.0,
            end_value=1.0,
            ramp_start_step=2,
            ramp_steps=4,
            curve_rate=0.0,
        )
        linear = LinearScheduler(0.0, 1.0, ramp_start_step=2, ramp_steps=4)
        for step in range(8):
            self.assertEqual(logarithmic.get_value(step), linear.get_value(step))

    def test_fixed_scheduler(self):
        scheduler = FixedScheduler(value=-0.5)
        self.assertEqual(scheduler.get_value(0), -0.5)
        self.assertEqual(scheduler.get_value(100), -0.5)

    def test_scheduler_config_builds_registered_schedulers(self):
        linear = SchedulerConfig(
            type="linear",
            start_value=1.0,
            end_value=0.0,
            ramp_start_step=3,
            ramp_steps=7,
        ).get_scheduler()
        logarithmic = SchedulerConfig(
            type="logarithmic",
            start_value=0.0,
            end_value=1.0,
            ramp_start_step=0,
            ramp_steps=10,
            curve_rate=4.0,
        ).get_scheduler()
        fixed = SchedulerConfig(type="fixed", value=3.0).get_scheduler()

        self.assertIsInstance(linear, LinearScheduler)
        self.assertIsInstance(logarithmic, LogarithmicScheduler)
        self.assertIsInstance(fixed, FixedScheduler)
        self.assertEqual(
            set(registered_schedulers), {"linear", "logarithmic", "fixed"}
        )


if __name__ == "__main__":
    unittest.main()
