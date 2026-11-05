"""Keep CLI-only flag validation out of spawned environment workers."""

import multiprocessing as mp
from pathlib import Path
import runpy
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _parse_environment_flags():
    # Spawn loads the entry module under this name, without running app.run().
    runpy.run_path(str(ROOT / "main.py"), run_name="__mp_main__")
    from absl import flags

    flags.FLAGS(["smacv2"])
    assert flags.FLAGS["config"].value is None


class TestCLIFlags(unittest.TestCase):
    def test_spawned_worker_can_parse_environment_flags(self):
        process = mp.get_context("spawn").Process(target=_parse_environment_flags)
        process.start()
        try:
            process.join(timeout=30)
            self.assertEqual(process.exitcode, 0)
        finally:
            if process.is_alive():
                process.kill()
                process.join()
            process.close()

    def test_cli_requires_config(self):
        result = subprocess.run(
            [sys.executable, "main.py", "train"], cwd=ROOT,
            capture_output=True, text=True, timeout=30,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Flag --config must have a value other than None", result.stderr)

    def test_cli_accepts_config_without_starting_training(self):
        # Exercise the actual CLI while mocking only the expensive training call.
        script = """
import runpy
import sys
from unittest.mock import patch

sys.argv = ['main.py', 'train', '--config=examples/smac_flat64_50v50_qplex_td1_long.yaml']
with patch('marlite.trainer.trainer_config.TrainerConfig') as trainer:
    try:
        runpy.run_path('main.py', run_name='__main__')
    except SystemExit as exc:
        assert exc.code in (None, 0), exc.code
    trainer.assert_called_once()
    trainer.return_value.run.assert_called_once_with()
"""
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=ROOT,
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
