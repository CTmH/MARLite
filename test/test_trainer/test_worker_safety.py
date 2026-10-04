"""Failures must stop training instead of leaving the parent or peers waiting."""

import os
import subprocess
import sys
from copy import deepcopy
from datetime import timedelta
import importlib
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from marlite.trainer import TrainerConfig
from marlite.trainer.trainer_worker_group.base_worker_group import BaseWorkerGroup, _slice_batch


class FaultWorker:
    def __init__(self, worker_id, fault, **kwargs):
        self.worker_id, self.fault = worker_id, fault
        if worker_id == 0 and fault == "INITIALIZE":
            raise ValueError("injected initialization error")

    def handle_command(self, cmd, param_queue, data_queue, loss_queue, ack_queue):
        if cmd == "STOP":
            return False
        if self.worker_id == 0 and cmd == self.fault:
            raise ValueError(f"injected {cmd} error")
        if self.worker_id == 0 and self.fault == "hard_exit":
            os._exit(7)
        if cmd == "TRAIN_STEP":
            data_queue.get()
            loss_queue.put({"loss": 1.})
        elif cmd == "SYNC_TO_MAIN":
            param_queue.put({})
        else:
            ack_queue.put("ACK")
        return True


class FaultGroup(BaseWorkerGroup):
    def _get_worker_class(self):
        return FaultWorker

    def _create_worker_kwargs(self):
        return {"fault": self.fault}


@pytest.mark.parametrize("fault", ["INITIALIZE", "TRAIN_STEP", "SYNC_TO_MAIN", "MOVE_TO_GPU", "hard_exit"])
def test_worker_failure_raises_and_cleans_peers(fault, caplog):
    group = FaultGroup([0, 1], 2)
    group.fault = fault
    processes = []
    try:
        with pytest.raises(RuntimeError, match="Distributed training aborted"):
            group.start_workers()
            processes = list(group.workers)
            if fault == "SYNC_TO_MAIN":
                group.read_params_from_worker0()
            elif fault == "MOVE_TO_GPU":
                group.move_models_to_gpu()
            else:
                group.train_step({"states": torch.ones(2, 1)})
        assert not group.workers
        assert all(not process.is_alive() for process in processes)
        assert "Distributed training aborted" in caplog.text
        assert ("exitcode=7" if fault == "hard_exit" else "injected") in caplog.text
    finally:
        group.shutdown()


@pytest.mark.parametrize("ranks", [1, 2, 3, 8])
@pytest.mark.parametrize("size", [1, 2, 3, 5, 64, 65, 68, 129])
def test_balanced_shards_cover_samples_and_keep_fields_aligned(size, ranks):
    batch = {"states": torch.arange(size), "graphs": list(range(size)),
             "labels": tuple(range(size)), "epoch": 7, "scale": torch.tensor(1.)}
    shards = _slice_batch(batch, ranks)
    lengths = [len(s["states"]) for s in shards]
    assert min(lengths) > 0 and max(lengths) - min(lengths) <= 1
    expected = batch["states"].repeat(ranks) if size < ranks else batch["states"]
    torch.testing.assert_close(torch.cat([s["states"] for s in shards]), expected)
    for shard in shards:
        assert shard["states"].tolist() == shard["graphs"] == list(shard["labels"])
        assert shard["epoch"] == 7 and shard["scale"] == 1
    shards[0]["states"][0] = -1
    assert batch["states"][0] == 0  # Worker tensors must not alias the input.


def test_empty_batch_still_rejected():
    with pytest.raises(ValueError, match="must be positive"):
        _slice_batch({"states": torch.empty(0, 2)}, 2)


def test_equal_shards_and_metadata():
    batch = {"states": torch.arange(8).reshape(4, 2), "graphs": [1, 2, 3, 4],
             "epoch": 7, "scale": torch.tensor(1.)}
    shards = _slice_batch(batch, 2)
    torch.testing.assert_close(torch.cat([s["states"] for s in shards]), batch["states"])
    assert shards[1]["graphs"] == [3, 4]
    assert shards[0]["epoch"] == 7
    with pytest.raises(ValueError, match="common sample count"):
        _slice_batch({"states": torch.zeros(4, 2), "graphs": [1, 2]}, 2)


@pytest.mark.parametrize("fault", ["TRAIN_STEP", "invalid_batch"])
def test_training_entry_exits_nonzero_without_orphan_workers(fault):
    # Leave the error uncaught, as main.py does. A leaked non-daemon worker
    # would keep this subprocess alive and trigger the timeout.
    script = f"""
from test.test_trainer.test_worker_safety import FaultGroup
import torch
if __name__ == '__main__':
    group = FaultGroup([0, 1], 2)
    group.fault = {fault!r}
    group.start_workers()
    group.train_step({{'states': torch.ones({0 if fault == 'invalid_batch' else 2}, 1)}})
"""
    result = subprocess.run([sys.executable, "-c", script],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert ("injected TRAIN_STEP error" if fault == "TRAIN_STEP" else "must be positive") in result.stderr


@pytest.mark.parametrize("module,name", [
    ("qplex_trainer", "TestQPLEXTrainer"),
    ("ae_group_consensus_trainer", "TestAEGroupConsensusTrainer"),
])
def test_first_epoch_rollback_keeps_learned_weights(tmp_path, module, name):
    fixture = getattr(importlib.import_module(f"test.test_trainer.test_{module}"), name)()
    fixture.setUp()
    config = deepcopy(fixture.config)
    config["trainer"].update(workdir=str(tmp_path), train_device="cpu", n_workers=0)
    trainer = TrainerConfig(config).create_trainer()
    snapshots = {}

    def learn(**kwargs):
        for name in ("eval_agent_group", "eval_critic", "ssl_model"):
            net = getattr(trainer, name, None)
            if net is not None:
                with torch.no_grad():
                    for parameter in net.parameters():
                        parameter.add_(1.)
                snapshots[name] = deepcopy(net.state_dict())
        return {"loss": 1., "critic_loss": 1., "ssl_loss": 1.}

    result = {metric: {"mean": 1.} for metric in trainer.eval_metric_list}
    with patch.object(trainer, "collect_experience"), patch.object(trainer, "learn", side_effect=learn), \
            patch.object(trainer, "evaluate", return_value=result):
        trainer.train(epochs=1, target_first_metric=2., rollback_interval=1)
    for name, state in snapshots.items():
        for key, value in getattr(trainer, name).state_dict().items():
            torch.testing.assert_close(value, state[key])


def _conditional_gradient_check(rank, rendezvous, backend):
    from marlite.trainer.trainer_worker.offpolicy_worker import OffPolicyWorker
    from marlite.trainer.trainer_worker.onpolicy_worker import OnPolicyWorker
    from marlite.trainer.trainer_worker.mappo_worker import MAPPOWorker
    from marlite.trainer.trainer_worker.qtran_worker import QTRANWorker

    device = f"cuda:{rank}" if backend == "nccl" else "cpu"
    if backend == "nccl":
        torch.cuda.set_device(rank)
    dist.init_process_group(backend, init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        for cls in (OffPolicyWorker, OnPolicyWorker, MAPPOWorker, QTRANWorker):
            worker = object.__new__(cls)
            for name in ("eval_agent_group", "eval_critic", "eval_v_net"):
                net = torch.nn.ParameterList([
                    torch.nn.Parameter(torch.ones(1, device=device)) for _ in range(3)
                ])
                # Different active heads on each rank; third head is globally unused.
                net[rank].grad = torch.full_like(net[rank], 2. * (rank + 1))
                setattr(worker, name, net)
            if cls is MAPPOWorker:
                worker._reduce_agent_gradients()
                worker._reduce_critic_gradients()
            else:
                worker.reduce_gradients()
            names = ["eval_agent_group", "eval_critic"]
            if cls is QTRANWorker:
                names.append("eval_v_net")
            for name in names:
                net = getattr(worker, name)
                torch.testing.assert_close(net[0].grad, torch.ones_like(net[0]))
                torch.testing.assert_close(net[1].grad, 2 * torch.ones_like(net[1]))
                assert net[2].grad is None
        # Real collectives must accept both uneven shards and singleton tails.
        from marlite.util.loss_mixer import reduce_mixed_gradients
        for size in (1, 3, 65):
            samples = torch.arange(1, size + 1, dtype=torch.float32, device=device)
            shards = _slice_batch({"samples": samples}, 2)
            parameter = torch.nn.Parameter(torch.ones((), device=device))
            (parameter * shards[rank]["samples"]).mean().backward()
            reduce_mixed_gradients([parameter])
            expected = torch.stack([s["samples"].mean() for s in shards]).mean()
            torch.testing.assert_close(parameter.grad, expected)
            if size == 65:
                # 33/32 shards: gradient = 33.25 rather than sample mean 33.
                torch.testing.assert_close(parameter.grad - samples.mean(),
                                           torch.tensor(.25, device=device))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("backend", ["gloo", "nccl"])
def test_ordinary_workers_conditional_gradients(tmp_path, backend):
    if backend == "nccl" and torch.cuda.device_count() < 2:
        pytest.skip("Two GPUs required")
    if backend == "gloo" and not dist.is_gloo_available():
        pytest.skip("Gloo unavailable")
    mp.spawn(_conditional_gradient_check,
             args=(f"file://{tmp_path / 'rendezvous'}", backend), nprocs=2)
