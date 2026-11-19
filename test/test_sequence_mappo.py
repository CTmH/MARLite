"""End-to-end sequence PPO tests without a simulator or legacy trainer fixtures."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from marlite.algorithm.sequence_agent_group import SequenceAgentGroup
from marlite.algorithm.sequence_value import SequenceValue
from marlite.algorithm.temporal import FrameEncoder, RecurrentGRU
from marlite.collection.parallel import ParallelCollector
from marlite.data import partition_episode
from marlite.learning.mappo import MAPPOLearner, prepare_ppo_targets
from marlite.training.mappo import SequenceMAPPOTrainer


class TinyParallel:
    """Deterministic ParallelEnv-shaped task with genuine death and timeout cases."""

    possible_agents = ["a", "b"]

    def __init__(self, limit=5, death=False, mask=False, missing_final=False):
        self.limit, self.death, self.mask = limit, death, mask
        self.missing_final = missing_final
        self.agents = []
        self.time = 0
        self.closed = False

    def action_space(self, _agent):
        return SimpleNamespace(n=3)

    def _obs(self, agents):
        result = {}
        for i, a in enumerate(self.possible_agents):
            if a in agents:
                obs = np.array([self.latent, self.time / 10, float(i)], dtype=np.float32)
                result[a] = (
                    {"observation": obs, "action_mask": np.array([1, 1, 0], dtype=bool)}
                    if self.mask
                    else obs
                )
        return result

    def reset(self, seed=None):
        self.latent = float(np.random.default_rng(seed).normal())
        self.time = 0
        self.agents = list(self.possible_agents)
        return self._obs(self.agents), {}

    def state(self):
        return np.array([self.latent, self.time / 10, float(len(self.agents))], dtype=np.float32)

    def step(self, actions):
        assert set(actions) == set(self.agents)
        before = list(self.agents)
        self.time += 1
        terminal = self.time >= self.limit
        terms = {a: terminal or (self.death and a == "a" and self.time >= 2) for a in before}
        rewards = {a: 1.0 + float(actions[a] == 1) for a in before}
        self.agents = [a for a in before if not terms[a]]
        observations = self._obs(self.agents)
        if self.missing_final and self.time == 2:
            self.agents = []
            observations = {}
        return observations, rewards, terms, {a: False for a in before}, {}

    def close(self):
        self.closed = True


def make_trainer(
    *,
    recurrent=True,
    factory=TinyParallel,
    max_steps=4,
    state_strategy="cached_burnin",
    vf_coef=0.5,
    workdir=None,
):
    torch.manual_seed(10)
    temporal = RecurrentGRU if recurrent else FrameEncoder
    actor = SequenceAgentGroup(nn.Linear(3, 8), temporal(8, 8), nn.Linear(8, 3))
    critic = SequenceValue(nn.Linear(3, 8), temporal(8, 8), nn.Linear(8, 1))
    learner = MAPPOLearner(
        actor,
        critic,
        torch.optim.Adam(actor.parameters(), lr=0.001),
        torch.optim.Adam(critic.parameters(), lr=0.001),
        burn_in=2,
        state_strategy=state_strategy,
        vf_coef=vf_coef,
    )
    return SequenceMAPPOTrainer(
        learner,
        ParallelCollector(factory, max_steps=max_steps),
        sequence_length=2,
        batch_sequences=2,
        ppo_epochs=2,
        workdir=workdir,
    )


def test_collection_death_timeout_and_actual_final_frame():
    trainer = make_trainer(factory=lambda: TinyParallel(death=True, mask=True))
    ep = trainer.collect(1)[0]
    assert ep.length == 4 and ep.frames["states"].shape == (5, 3)
    assert ep.transitions["truncated"][-1] and not ep.transitions["terminated"][-1]
    assert ep.frames["states"][-1, 1].item() == pytest.approx(0.4)
    assert ep.frames["alive"][:, 0].tolist() == [True, True, False, False, False]
    assert ep.transitions["rewards"][1, 0] > 0  # Death-transition reward retained.
    assert ep.transitions["rewards"][2, 0] == 0
    assert ep.state_snapshots["actor_state.h"].shape == (1, 2, 1, 8)
    assert ep.state_snapshots["critic_state.h"].shape == (1, 1, 1, 8)
    assert ep.return_masks()[0][-1]
    trainer = make_trainer(max_steps=8)
    terminal = trainer.collect(1)[0]
    assert terminal.transitions["terminated"][-1]
    assert not terminal.return_masks()[0][-1]
    broken = make_trainer(factory=lambda: TinyParallel(missing_final=True))
    with pytest.raises(ValueError, match="final observation"):
        broken.collect(1)


@pytest.mark.parametrize("recurrent", [False, True])
@pytest.mark.parametrize("strategy", ["cached_burnin", "prefix_recompute"])
def test_ppo_behavior_replay_ratio_one_and_fixed_labels(recurrent, strategy):
    trainer = make_trainer(recurrent=recurrent, state_strategy=strategy)
    episodes = trainer.collect(2)
    before = deepcopy(episodes[0].transitions)
    prepared = prepare_ppo_targets(episodes)
    assert "advantages" not in episodes[0].transitions
    sources = [s for i, ep in prepared.items() for s in partition_episode(i, ep, length=2)]
    _, _, metrics = trainer.learner.losses(prepared, sources)
    assert abs(metrics["approx_kl"]) < 1e-6
    assert metrics["clip_fraction"] == 0
    assert metrics["agent_steps"] == 16 and metrics["team_steps"] == 8
    labels = prepared[0].transitions["value_targets"].clone()
    original = [p.clone() for p in trainer.learner.actor.parameters()]
    result = trainer.learner.update(prepared, sources)
    assert all(np.isfinite(v) for v in result.values())
    assert any(
        not torch.equal(p, old) for p, old in zip(trainer.learner.actor.parameters(), original)
    )
    assert torch.equal(labels, prepared[0].transitions["value_targets"])
    for key in before:
        torch.testing.assert_close(before[key], episodes[0].transitions[key])


def test_hand_computed_stateless_objective_and_disabled_value_optimizer():
    trainer = make_trainer(recurrent=False, vf_coef=0.0)
    learner = trainer.learner
    episodes = prepare_ppo_targets(trainer.collect(1), normalize=False)
    ep = episodes[0]
    sources = partition_episode(0, ep, length=4)
    policy_loss, critic_loss, metrics = learner.losses(episodes, sources)
    expected = -ep.transitions["advantages"].mean() - learner.entropy_coef * metrics["entropy"]
    torch.testing.assert_close(policy_loss, expected)
    assert critic_loss == 0
    before = deepcopy(learner.critic.state_dict())
    learner.update(episodes, sources)
    assert learner.critic_updates == 0 and not learner.critic_optimizer.state
    for key, value in learner.critic.state_dict().items():
        torch.testing.assert_close(value, before[key])
    bad = deepcopy(episodes)
    bad[1] = deepcopy(ep)
    bad[1].policy_version += 1
    with pytest.raises(ValueError, match="versions"):
        learner.losses(bad, [sources[0], type(sources[0])(1, 0, 4)])


def test_training_and_evaluation_are_independent(tmp_path):
    trainer = make_trainer(workdir=tmp_path)
    episodes = trainer.collect(2)
    before = {k: getattr(trainer, k) for k in ("env_steps", "agent_steps", "collection_round")}
    rng = torch.get_rng_state().clone()
    evaluated = trainer.evaluate(2)
    assert torch.equal(rng, torch.get_rng_state())
    assert evaluated["episode_seeds"] != [ep.seed for ep in episodes.values()]
    assert before == {k: getattr(trainer, k) for k in before}
    trainer.run(2, train_episodes=2, eval_episodes=2, checkpoint_interval=1)
    assert trainer.iterations == 2 and trainer.policy_version == 2
    assert trainer.learner.actor_updates == 8
    assert (tmp_path / "checkpoints" / "best.pt").exists()
    assert (tmp_path / "checkpoints" / "iteration_2.pt").exists()
    events = (tmp_path / "events.jsonl").read_text().splitlines()
    assert len(events) == 2 and '"evaluation"' in events[0]


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


def test_complete_checkpoint_matches_uninterrupted_training(tmp_path):
    original = make_trainer()
    original.train_iteration(2)
    path = tmp_path / "resume.pt"
    original.save_checkpoint(path)
    expected_metrics = original.train_iteration(2)
    expected = deepcopy(original.learner.state_dict())
    restored = make_trainer()
    restored.load_checkpoint(path)
    actual_metrics = restored.train_iteration(2)
    assert actual_metrics == expected_metrics
    assert_tree_equal(expected, restored.learner.state_dict())
    assert restored.env_steps == original.env_steps
    assert restored.agent_steps == original.agent_steps
    assert restored.collection_round == original.collection_round
    incompatible = make_trainer(state_strategy="prefix_recompute")
    with pytest.raises(ValueError, match="specification"):
        incompatible.load_checkpoint(path)


def test_reappearing_agent_resets_state():
    class Respawn(TinyParallel):
        def step(self, actions):
            result = super().step(actions)
            observations, rewards, terms, truncs, infos = result
            if self.time == 3:
                self.agents = ["a", "b"]
                observations = self._obs(self.agents)
            return observations, rewards, terms, truncs, infos

    trainer = make_trainer(factory=lambda: Respawn(death=True), max_steps=4)
    ep = trainer.collect(1)[0]
    assert ep.frames["alive"][:4, 0].tolist() == [True, True, False, True]
    assert ep.frames["agent_reset"][:4, 0].tolist() == [True, False, False, True]
    prepared = prepare_ppo_targets({0: ep})
    _, _, metrics = trainer.learner.losses(prepared, partition_episode(0, ep, length=2))
    assert abs(metrics["approx_kl"]) < 1e-6


def test_real_mpe2_short_training(tmp_path):
    pytest.importorskip("mpe2")
    from mpe2 import simple_spread_v3

    def factory():
        return simple_spread_v3.parallel_env(N=2, max_cycles=3, continuous_actions=False)

    env = factory()
    try:
        obs, _ = env.reset(seed=1)
        obs_dim = int(np.prod(next(iter(obs.values())).shape))
        state_dim = int(np.prod(env.state().shape))
        action_dim = env.action_space(env.possible_agents[0]).n
    finally:
        env.close()
    actor = SequenceAgentGroup(
        nn.Sequential(nn.Flatten(1), nn.Linear(obs_dim, 8)),
        RecurrentGRU(8, 8),
        nn.Linear(8, action_dim),
    )
    critic = SequenceValue(
        nn.Sequential(nn.Flatten(1), nn.Linear(state_dim, 8)), RecurrentGRU(8, 8), nn.Linear(8, 1)
    )
    learner = MAPPOLearner(
        actor,
        critic,
        torch.optim.Adam(actor.parameters(), lr=0.001),
        torch.optim.Adam(critic.parameters(), lr=0.001),
        burn_in=1,
    )
    trainer = SequenceMAPPOTrainer(
        learner,
        ParallelCollector(factory, max_steps=3),
        sequence_length=2,
        batch_sequences=2,
        ppo_epochs=1,
        workdir=tmp_path,
    )
    trainer.run(1, train_episodes=2, eval_episodes=2, checkpoint_interval=1)
    assert trainer.env_steps == 6 and learner.actor_updates == 2


def test_collector_uses_current_info_masks_when_observations_are_plain_arrays():
    class InfoMaskEnv(TinyParallel):
        def infos(self):
            return {a: {"action_mask": [False, True, False]} for a in self.agents}

        def reset(self, seed=None):
            obs, _ = super().reset(seed)
            return obs, self.infos()

        def step(self, actions):
            assert all(action == 1 for action in actions.values())
            obs, rewards, terms, truncs, _ = super().step(actions)
            return obs, rewards, terms, truncs, self.infos()

    trainer = make_trainer(factory=InfoMaskEnv)
    episode = trainer.collector.collect_episode(
        trainer.learner.actor, trainer.learner.critic, seed=7
    )
    assert episode.transitions["actions"].eq(1).all()
    assert not episode.frames["available_actions"][..., 0].any()
