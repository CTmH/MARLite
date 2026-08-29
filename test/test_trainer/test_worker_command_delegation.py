"""Regression tests for generic worker-command delegation."""

from queue import Queue
import unittest

from marlite.trainer.trainer_worker.g2anet_mappo_worker import (
    G2ANetMAPPOWorker,
)
from marlite.trainer.trainer_worker.ssl_gc_mappo_worker import (
    SSLGroupConsensusMAPPOWorker,
)


class TestWorkerCommandDelegation(unittest.TestCase):
    """Specialised workers must preserve the BaseWorker command protocol."""

    def _assert_average_command_is_delegated(self, worker_cls, command, method_name):
        # Construct without model setup: the generic command only needs the
        # synchronisation hook and an acknowledgement queue.
        worker = worker_cls.__new__(worker_cls)
        calls = []
        setattr(worker, method_name, lambda: calls.append(method_name))
        ack_queue = Queue()

        should_continue = worker.handle_command(
            command,
            Queue(),
            Queue(),
            Queue(),
            ack_queue,
        )

        self.assertTrue(should_continue)
        self.assertEqual(calls, [method_name])
        self.assertEqual(ack_queue.get_nowait(), "ACK")

    def test_specialised_workers_delegate_eval_parameter_averaging(self):
        for worker_cls in (SSLGroupConsensusMAPPOWorker, G2ANetMAPPOWorker):
            with self.subTest(worker=worker_cls.__name__):
                self._assert_average_command_is_delegated(
                    worker_cls,
                    "AVERAGE_EVAL_PARAMS",
                    "synchronize_eval_params",
                )

    def test_specialised_workers_delegate_target_parameter_averaging(self):
        for worker_cls in (SSLGroupConsensusMAPPOWorker, G2ANetMAPPOWorker):
            with self.subTest(worker=worker_cls.__name__):
                self._assert_average_command_is_delegated(
                    worker_cls,
                    "AVERAGE_TARGET_PARAMS",
                    "synchronize_target_params",
                )
