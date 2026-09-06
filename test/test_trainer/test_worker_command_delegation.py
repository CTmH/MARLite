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

    def test_ssl_mappo_worker_keeps_joint_and_adds_separate_train_commands(self):
        worker = SSLGroupConsensusMAPPOWorker.__new__(
            SSLGroupConsensusMAPPOWorker
        )
        calls = []
        worker.train_step = (
            lambda batch: calls.append(("joint", batch))
            or {"loss": 1.0, "critic_loss": 2.0, "ssl_loss": 3.0}
        )
        worker._ppo_train_step = (
            lambda batch: calls.append(("ppo", batch))
            or {"loss": 1.0, "rl_loss": 1.0}
        )
        worker._ssl_train_step = (
            lambda batch: calls.append(("ssl", batch))
            or {"loss": 2.0, "ssl_loss": 2.0}
        )

        for command, expected_phase, expected_result in (
            (
                "TRAIN_STEP",
                "joint",
                {"loss": 1.0, "critic_loss": 2.0, "ssl_loss": 3.0},
            ),
            ("PPO_TRAIN_STEP", "ppo", {"loss": 1.0, "rl_loss": 1.0}),
            ("SSL_TRAIN_STEP", "ssl", {"loss": 2.0, "ssl_loss": 2.0}),
        ):
            batch = {"data": command}
            data_queue = Queue()
            loss_queue = Queue()
            data_queue.put(batch)

            self.assertTrue(
                worker.handle_command(
                    command, Queue(), data_queue, loss_queue, Queue()
                )
            )
            self.assertEqual(calls[-1], (expected_phase, batch))
            self.assertEqual(loss_queue.get_nowait(), expected_result)

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

    def test_ssl_mappo_worker_group_sends_phase_as_command_not_batch_data(self):
        group = SSLGroupConsensusMAPPOWorkerGroup.__new__(
            SSLGroupConsensusMAPPOWorkerGroup
        )
        group.world_size = 2
        group.cmd_queues = [Queue(), Queue()]
        group.data_queues = [Queue(), Queue()]
        group.loss_queue = Queue()
        group.param_queues = [Queue(), Queue()]
        group.ack_queues = [Queue(), Queue()]
        batch = {"data": 7}

        for method_name, command, result in (
            (
                "train_step",
                "TRAIN_STEP",
                {"loss": 1.0, "critic_loss": 2.0, "ssl_loss": 3.0},
            ),
            (
                "ppo_train_step",
                "PPO_TRAIN_STEP",
                {"loss": 1.0, "rl_loss": 1.0},
            ),
            (
                "ssl_train_step",
                "SSL_TRAIN_STEP",
                {"loss": 2.0, "ssl_loss": 2.0},
            ),
        ):
            group.loss_queue.put(result)
            group.loss_queue.put(result)
            actual = getattr(group, method_name)(batch)

            self.assertEqual(actual, result)
            for command_queue, data_queue in zip(
                group.cmd_queues, group.data_queues
            ):
                self.assertEqual(command_queue.get_nowait(), command)
                self.assertEqual(data_queue.get_nowait(), batch)
            self.assertNotIn("training_phase", batch)

        for ack_queue in group.ack_queues:
            ack_queue.put("ACK")
        group.set_training_epoch(9)
        for command_queue, param_queue in zip(
            group.cmd_queues, group.param_queues
        ):
            self.assertEqual(command_queue.get_nowait(), "SET_TRAINING_EPOCH")
            self.assertEqual(param_queue.get_nowait(), 9)
