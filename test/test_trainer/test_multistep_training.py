"""Small real-model updates, without SC2, subprocesses, or GPU requirements."""

from copy import deepcopy
import importlib
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from marlite.trainer.trainer_config import TrainerConfig
from marlite.trainer.trainer_worker_group.base_worker_group import _slice_batch
from marlite.util.trajectory_dataset import TrajectoryDataLoader
from marlite.util.return_estimation import ppo_targets
from marlite.util.serialization import serialize_to_buffer, get_state_dict
from marlite.util.trajectory_dataset import GroupSSLEnrichedTrajectoryDataset, GraphSSLEnrichedTrajectoryDataset
from test.test_trainer import test_mappo_trainer, test_qmix_trainer


def small_episode(matrix=False, terminal=False):
    agents = [f"agent_{i}" for i in range(3)]
    rng = np.random.default_rng(42)
    obs = [{a: rng.normal(size=18).astype(np.float32) for a in agents} for _ in range(6)]
    states = [rng.normal(size=(3, 18) if matrix else 54).astype(np.float32) for _ in range(6)]
    alive = [{a: (i != 0 or t == 0) and not (terminal and t == 5)
              for i, a in enumerate(agents)} for t in range(6)]
    groups = [{a: (i if alive[t][a] else -1) for i, a in enumerate(agents)} for t in range(6)]
    masks = [{a: np.array([True, True, False, False, False]) for a in agents} for _ in range(6)]
    episode = {
        "states": states[:-1], "next_states": states[1:],
        "observations": obs[:-1], "next_observations": obs[1:],
        "alive_mask": alive[:-1], "next_alive_mask": alive[1:],
        "avail_actions": masks[:-1], "next_avail_actions": masks[1:],
        "group_indices": groups[:-1], "next_group_indices": groups[1:],
        "edge_indices": [np.array([[0, 1, 2], [0, 1, 2]]) for _ in range(5)],
        "next_edge_indices": [np.array([[0, 1, 2], [0, 1, 2]]) for _ in range(5)],
        "actions": [{a: 0 for a in agents} for _ in range(5)],
        "rewards": [{a: float(alive[t][a]) for a in agents} for t in range(5)],
        "terminations": [{a: not alive[t+1][a] for a in agents} for t in range(5)],
        "truncations": [{a: not alive[t+1][a] or t == 4 for a in agents} for t in range(5)],
        "all_log_probs": [{a: -np.log(2.) for a in agents} for _ in range(5)],
        "log_probs": [{a: -np.log(2.) for a in agents} for _ in range(5)],
        "all_agents_sum_rewards": [3., 2., 2., 2., 2.],
        "infos": [{} for _ in range(5)],
        "episode_length": 5, "episode_reward": 11., "win_tag": terminal,
    }
    return episode


def attention(input_dim, output_dim):
    return dict(model_type="SimpleResAttSeqEnc", input_dim=input_dim, embed_dim=16,
                output_dim=output_dim, num_heads=2, max_seq_len=7, dropout=0.)


@pytest.mark.parametrize("onpolicy", [True, False])
@pytest.mark.parametrize("sequence", [True, False])
@pytest.mark.parametrize("matrix", [True, False])
def test_attention_updates_with_critic_variants(onpolicy, sequence, matrix):
    fixture = test_mappo_trainer.TestMAPPOTrainer() if onpolicy else test_qmix_trainer.TestQMixTrainer()
    fixture.setUp()
    config = deepcopy(fixture.config)
    config["trainer"].update(train_device="cpu", n_workers=0)
    config["replay_buffer"]["traj_len"] = 2  # n_steps deliberately exceeds history.
    config["agent_group"]["models"]["model1"]["encoder"] = attention(18, 32)
    if onpolicy:
        config["trainer"].update(advantage_estimator="gae", gae_lambda=.9, normalize_advantages=True)
    else:
        config["trainer"]["n_steps"] = 3
    critic = config["critic"]
    dim = 54
    if matrix:
        dim = 16
        critic["feature_extractor"] = dict(model_type="SimpleResAttMaskedStateEnc",
            input_dim=18, embed_dim=dim, num_heads=2, max_seq_len=3, dropout=0.)
    if sequence:
        critic["type"] = "SeqMAPPOCritic" if onpolicy else "SeqQMixer"
        critic["seq_model"] = attention(dim, 16)
        dim = 16
    if onpolicy:
        critic["model"]["layers"][0]["in_features"] = dim
    else:
        critic["model"]["state_shape"] = dim
    with tempfile.TemporaryDirectory() as directory:
        config["trainer"]["workdir"] = directory
        trainer = TrainerConfig(config).create_trainer()
        for terminal in (True, False):
            trainer.replaybuffer.add_episode(small_episode(matrix, terminal))
        if onpolicy:
            ds = trainer._prepare_ppo_dataset(10, 3)
            batch = next(iter(TrajectoryDataLoader(ds, 10, shuffle=False, num_workers=0)))
            assert batch["advantages"].mean().abs() < 1e-6
            # Labels are detached, fixed and shared across worker slices.
            targets = batch["value_targets"].clone()
            with torch.no_grad():
                for parameter in trainer.eval_critic.parameters():
                    parameter.add_(.01)
            adv, ret = ppo_targets(batch, torch.randn(10), None, trainer.gamma, None)
            torch.testing.assert_close(ret.double(), targets.double())
            slices = _slice_batch(batch, 2)
            torch.testing.assert_close(torch.cat([s["advantages"] for s in slices]).to(adv), adv)
            with patch.object(trainer, "_prepare_ppo_dataset", return_value=ds) as prepare:
                metrics = trainer.learn(10, 4, times=2)
                assert prepare.call_count == 1
        else:
            ds = trainer._sample_training_data(10)
            assert ds.n_steps == 3
            metrics = trainer.learn(10, 4, times=2)
        assert all(np.isfinite(value) for value in metrics.values())


def test_gae_labels_match_independent_full_episode_calculation():
    fixture = test_mappo_trainer.TestMAPPOTrainer()
    fixture.setUp()
    fixture.config["trainer"].update(advantage_estimator="gae", gae_lambda=.7, n_workers=0)
    with tempfile.TemporaryDirectory() as directory:
        trainer = fixture._create_trainer(directory)
        trainer.replaybuffer.add_episode(small_episode(terminal=False))
        trainer.eval_critic.eval()
        ep = trainer.replaybuffer.episode_buffer[0]
        with torch.no_grad():
            v = trainer.eval_critic(torch.tensor(np.array(ep["states"])).unsqueeze(1),
                torch.tensor([list(a.values()) for a in ep["alive_mask"]]).unsqueeze(1),
                torch.zeros(5, 1, dtype=torch.bool))["v"].flatten()
            nv = trainer.eval_critic(torch.tensor(np.array(ep["next_states"])).unsqueeze(1),
                torch.tensor([list(a.values()) for a in ep["next_alive_mask"]]).unsqueeze(1),
                torch.zeros(5, 1, dtype=torch.bool))["v"].flatten()
        ds = trainer._prepare_ppo_dataset(5, 2)
        tail = 0.
        for t in reversed(range(5)):
            delta = sum(ep["rewards"][t].values()) + trainer.gamma * nv[t].item() - v[t].item()
            tail = delta + trainer.gamma * trainer.gae_lambda * tail
            assert ds.training_targets[0, t]["advantages"] == pytest.approx(tail, abs=2e-5)


@pytest.mark.parametrize("module_name,fixture_name,worker_module,worker_name", [
    ("mappo", "TestMAPPOTrainer", "mappo", "MAPPOWorker"),
    ("g2anet_mappo", "TestGraphMAPPOTrainer", "g2anet_mappo", "G2ANetMAPPOWorker"),
    ("vaegc_mappo", "TestVAEGCMAPPOTrainer", "ssl_gc_mappo", "SSLGroupConsensusMAPPOWorker"),
    ("ae_gc_mappo", "TestAEGCMAPPOTrainer", "ssl_gc_mappo", "SSLGroupConsensusMAPPOWorker"),
    ("qmix", "TestQMixTrainer", "qmix", "QMIXWorker"),
    ("qplex", "TestQPLEXTrainer", "qplex", "QPLEXWorker"),
    ("qtran", "TestQTRANTrainer", "qtran", "QTRANWorker"),
    ("graph_qmix", "TestGraphQMIXTrainer", "graph", "GraphWorker"),
    ("group_consensus", "TestGroupConsensusTrainer", "group_consensus", "GroupConsensusWorker"),
    ("msg_aggr_qmix", "TestMsgAggrQMIXTrainer", "msg_aggr", "MsgAggrWorker"),
    ("msg_aggr_qmix", "TestMsgAggrQMIXTrainer", "msg_aggr", "ProbMsgAggrWorker"),
    ("vae_group_consensus", "TestGroupConsensusTrainer", "ssl_group_consensus", "SSLGroupConsensusWorker"),
    ("ae_group_consensus", "TestAEGroupConsensusTrainer", "ssl_group_consensus", "SSLGroupConsensusWorker"),
    ("self_supervised_gnn", "TestVAEGraphQMIXBattle", "vae_graph", "VAEGraphQMIXWorker"),
])
def test_family_trainer_and_worker(module_name, fixture_name, worker_module, worker_name):
    fixture = getattr(importlib.import_module(f"test.test_trainer.test_{module_name}_trainer"
        if module_name != "self_supervised_gnn" else "test.test_trainer.test_self_supervised_gnn"), fixture_name)()
    fixture.setUp()
    config = deepcopy(fixture.config)
    onpolicy = "mappo" in module_name
    if worker_name == "ProbMsgAggrWorker":
        config["trainer"]["type"] = "ProbMsgAggr"
        config["agent_group"]["type"] = "ProbObsMsgAggr"
        config["agent_group"]["aggr_model_config"]["layers"][-2]["out_features"] = 64
        config["critic"]["type"] = "ProbQMixer"
        config["critic"]["feature_extractor"]["layers"][-2]["out_features"] = 64
    if module_name == "vaegc_mappo":
        # The small legacy MPE fixture only exercises RL warmup and has no
        # group-aware SSL constructor. Use the valid AE fixture for joint VAE.
        ae = importlib.import_module("test.test_trainer.test_ae_gc_mappo_trainer").TestAEGCMAPPOTrainer()
        ae.setUp()
        config = deepcopy(ae.config)
        config["agent_group"]["consensus_mode"] = "vae"
        config["agent_group"]["models"]["model_0"]["group_estimate_feature_extractor"]["layers"][-1]["out_features"] = 128
        config["trainer"].update(consensus_mode="vae", kl_on_agent=True, kl_on_group=True,
                                  kl_divergence_weight=.001)
    config["trainer"].update(train_device="cpu", n_workers=0)
    if onpolicy:
        config["trainer"].update(advantage_estimator="gae", gae_lambda=.9)
        config["trainer"].pop("warmup_iterations", None)
    else:
        config["trainer"]["n_steps"] = 3
        config["trainer"].pop("warmup_epochs", None)
    if "self_supervised_learning" in config:
        config["self_supervised_learning"]["data_constructor"]["n_workers"] = 0
    with tempfile.TemporaryDirectory() as directory:
        config["trainer"]["workdir"] = directory
        trainer = TrainerConfig(config).create_trainer()
        for name in ("warmup_iterations", "warmup_epochs"):
            if hasattr(trainer, name):
                setattr(trainer, name, 0)
        rollout = importlib.import_module("marlite.rollout.multiprocess_rollout")
        params = serialize_to_buffer(get_state_dict(trainer.eval_agent_group))
        # Exercise real environments/models in-process; only replace IPC storage.
        with patch.object(rollout, "SharedMemory", return_value=SimpleNamespace(buf=params, close=lambda: None)):
            ep = rollout.multiprocess_rollout(trainer.env_config, trainer.agent_group_config,
                ("in-process", len(params)), rnn_traj_len=config["replay_buffer"]["traj_len"],
                episode_limit=4, epsilon=1. if onpolicy else .5)
        trainer.replaybuffer.add_episode(ep)
        assert ep["episode_length"] == 4
        terminal_ep = deepcopy(ep)
        terminal_ep["terminations"][-1] = {a: True for a in ep["terminations"][-1]}
        terminal_ep["next_alive_mask"][-1] = {a: False for a in ep["next_alive_mask"][-1]}
        terminal_ep["next_group_indices"][-1] = {a: -1 for a in ep["next_group_indices"][-1]}
        trainer.replaybuffer.add_episode(terminal_ep)
        metrics = trainer.learn(8, 2, times=1)
        assert all(np.isfinite(v) for v in metrics.values())
        ds = trainer._prepare_ppo_dataset(4, 2) if onpolicy else trainer._sample_training_data(4)
        if hasattr(trainer, "data_constructor"):
            cls = GraphSSLEnrichedTrajectoryDataset if worker_module == "vae_graph" else GroupSSLEnrichedTrajectoryDataset
            ds = cls(ds, trainer.data_constructor)
        batch = next(iter(TrajectoryDataLoader(ds, 4, shuffle=False, num_workers=0)))
        worker_cls = getattr(importlib.import_module(f"marlite.trainer.trainer_worker.{worker_module}_worker"), worker_name)
        worker = worker_cls.__new__(worker_cls)
        worker.__dict__.update(trainer.__dict__)
        worker.device = torch.device("cpu")
        worker.current_training_epoch = trainer.current_epoch
        if worker_name == "ProbMsgAggrWorker":
            worker.Normal = torch.distributions.Normal
            worker.kl_divergence = torch.distributions.kl_divergence
        worker.reduce_gradients = lambda: None
        worker._reduce_agent_gradients = lambda: None
        worker._reduce_critic_gradients = lambda: None
        worker_metrics = worker.train_step(batch)
        assert all(np.isfinite(v) for v in worker_metrics.values())
