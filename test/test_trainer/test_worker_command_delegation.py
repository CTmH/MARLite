"""Regression tests for generic worker-command delegation."""

from queue import Queue
import unittest

from marlite.trainer.trainer_worker.g2anet_mappo_worker import (
    G2ANetMAPPOWorker,
)
from marlite.trainer.trainer_worker.ssl_gc_mappo_worker import (
    SSLGroupConsensusMAPPOWorker,
)
from marlite.trainer.trainer_worker_group.ssl_gc_mappo_worker_group import (
    SSLGroupConsensusMAPPOWorkerGroup,
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

    def test_ssl_mappo_worker_keeps_joint_train_step_and_epoch_command(self):
        worker = SSLGroupConsensusMAPPOWorker.__new__(
            SSLGroupConsensusMAPPOWorker
        )
        batch = {"data": "joint"}
        worker.train_step = lambda value: {
            "loss": value["data"],
            "critic_loss": 2.0,
            "ssl_loss": 3.0,
        }
        data_queue = Queue()
        loss_queue = Queue()
        data_queue.put(batch)

        self.assertTrue(
            worker.handle_command(
                "TRAIN_STEP", Queue(), data_queue, loss_queue, Queue()
            )
        )
        self.assertEqual(loss_queue.get_nowait()["critic_loss"], 2.0)

        epoch_queue = Queue()
        ack_queue = Queue()
        epoch_queue.put(12)
        self.assertTrue(
            worker.handle_command(
                "SET_TRAINING_EPOCH",
                epoch_queue,
                Queue(),
                Queue(),
                ack_queue,
            )
        )
        self.assertEqual(worker.current_training_epoch, 12)
        self.assertEqual(ack_queue.get_nowait(), "ACK")

    def test_ssl_mappo_worker_group_sends_joint_train_command(self):
        group = SSLGroupConsensusMAPPOWorkerGroup.__new__(
            SSLGroupConsensusMAPPOWorkerGroup
        )
        group.world_size = 2
        group.cmd_queues = [Queue(), Queue()]
        group.data_queues = [Queue(), Queue()]
        group.loss_queue = Queue()
        batch = {"data": 7}
        result = {"loss": 1.0, "critic_loss": 2.0, "ssl_loss": 3.0}

        group.loss_queue.put(result)
        group.loss_queue.put(result)
        self.assertEqual(group.train_step(batch), result)

        for command_queue, data_queue in zip(
            group.cmd_queues, group.data_queues
        ):
            self.assertEqual(command_queue.get_nowait(), "TRAIN_STEP")
            self.assertEqual(data_queue.get_nowait(), batch)
