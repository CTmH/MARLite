"""Separate mathematical and integration checks for QMIX, QPLEX and QTRAN."""

from copy import deepcopy

import pytest
import torch
from torch import nn

from marlite.algorithm.sequence_agent_group import SequenceAgentGroup, PolicyOutput
from marlite.algorithm.sequence_mixers import (
    MonotonicMixing,
    DuplexTransformation,
    JointAdvantageWeights,
    SequenceQMixer,
    SequenceDuplexMixer,
    CounterfactualJoint,
    SequenceCounterfactualMixer,
)
from marlite.algorithm.sequence_value import TeamSequenceEncoder, SequenceValue
from marlite.algorithm.temporal import FrameEncoder, RecurrentGRU
from marlite.collection.parallel import ParallelCollector
from marlite.data import partition_episode, collate_sequences, SequenceSlice
from marlite.data.replay import EpisodeReplay
from marlite.learning.value import (
    ValueLearner,
    QMIXObjective,
    QPLEXObjective,
    QTRANObjective,
    nstep_targets,
)
from test.test_sequence_mappo import TinyParallel, assert_tree_equal


def make_value_learner(
    algorithm,
    *,
    recurrent=True,
    n_steps=3,
    target_mode="hard",
    target_interval=2,
    state_strategy="zero_burnin",
    obs_dim=3,
    state_dim=3,
    action_dim=3,
    n_agents=2,
):
    torch.manual_seed(21)
    temporal = RecurrentGRU if recurrent else FrameEncoder
    actor = SequenceAgentGroup(
        nn.Linear(obs_dim, 8), temporal(8, 8), nn.Linear(8, action_dim), "q_values"
    )
    state_encoder = TeamSequenceEncoder(nn.Linear(state_dim, 8), temporal(8, 8))
    if algorithm == "qmix":
        mixing = MonotonicMixing(
            n_agents,
            4,
            nn.Linear(8, n_agents * 4),
            nn.Linear(8, 4),
            nn.Linear(8, 4),
            nn.Linear(8, 1),
        )
        critic = SequenceQMixer(state_encoder, mixing)
        objective = QMIXObjective()
    elif algorithm == "qplex":
        transform = DuplexTransformation(
            n_agents,
            4,
            [nn.Linear(8, 4) for _ in range(2)],
            [nn.Linear(8, n_agents * 4) for _ in range(2)],
            nn.Linear(8, 1),
            head_weight=nn.Linear(8, 2),
            regularization=0.01,
        )
        attention = JointAdvantageWeights(
            [nn.Linear(8, 1) for _ in range(2)],
            [nn.Linear(8, n_agents) for _ in range(2)],
            [nn.Linear(8 + n_agents * action_dim, n_agents) for _ in range(2)],
        )
        critic = SequenceDuplexMixer(
            state_encoder,
            transform,
            attention,
            nn.Linear(8, n_agents),
            nn.Linear(8, n_agents),
            action_dim=action_dim,
            n_agents=n_agents,
        )
        objective = QPLEXObjective()
    else:
        core = CounterfactualJoint(
            nn.Linear(8 + action_dim, 8), nn.Linear(8, 8), nn.Linear(8, action_dim), action_dim
        )
        critic = SequenceCounterfactualMixer(core)
        value = SequenceValue(nn.Linear(state_dim, 8), temporal(8, 8), nn.Linear(8, 1))
        objective = QTRANObjective(value, lambda_opt=0.3, lambda_nopt=0.7)
    return ValueLearner(
        actor,
        critic,
        objective,
        torch.optim.Adam(actor.parameters(), lr=0.001),
        torch.optim.Adam(critic.parameters(), lr=0.001),
        objective_optimizer=(
            torch.optim.Adam(objective.parameters(), lr=0.001)
            if objective.value_network is not None
            else None
        ),
        n_steps=n_steps,
        burn_in=2,
        target_mode=target_mode,
        target_interval=target_interval,
        state_strategy=state_strategy,
    )


def collect_values(learner, death=True, max_steps=4, epsilon=0.5):
    collector = ParallelCollector(lambda: TinyParallel(death=death, mask=True), max_steps=max_steps)
    return {
        i: collector.collect_episode(learner.actor, seed=50 + i, epsilon=epsilon, policy_version=i)
        for i in range(2)
    }


@pytest.mark.parametrize("gamma", [0.0, 0.9, 1.0])
@pytest.mark.parametrize("horizon", [1, 3, 8])
@pytest.mark.parametrize("terminated", [False, True])
def test_nstep_matches_independent_scalar_endpoints(gamma, horizon, terminated):
    learner = make_value_learner("qmix", n_steps=horizon)
    episodes = collect_values(learner, death=False, max_steps=8 if terminated else 4)
    ep = episodes[0]
    sources = partition_episode(0, ep, length=2, burn_in=0, future=horizon - 1)
    batch = collate_sequences(episodes, sources)
    joint = torch.arange(batch.filled.shape[1] + 1).float()[None].expand(len(sources), -1)
    target, endpoint, discount = nstep_targets(batch, joint, gamma, horizon)
    for row, source in enumerate(sources):
        begin = source.start
        for col in range(min(2, ep.length - begin)):
            position = begin + col
            steps = min(horizon, ep.length - position)
            expected = sum(
                gamma**k * float(ep.transitions["rewards"][position + k].sum())
                for k in range(steps)
            )
            end = col + steps
            should_bootstrap = not bool(ep.transitions["terminated"][position + steps - 1])
            expected += gamma**steps * float(joint[row, end]) * should_bootstrap
            assert target[row, col] == pytest.approx(expected)
            assert endpoint[row, col] == end
            assert discount[row, col] == pytest.approx(gamma**steps * should_bootstrap)


def test_nstep_rejects_missing_future_and_ignores_terminal_nan():
    learner = make_value_learner("qmix")
    episodes = collect_values(learner, death=False)
    batch = collate_sequences(episodes, [SequenceSlice(0, 0, 1)])
    with pytest.raises(ValueError, match="future tail"):
        nstep_targets(batch, torch.ones(1, 2), 0.9, 3)
    terminal = collect_values(learner, death=False, max_steps=8)
    ep = terminal[0]
    batch = collate_sequences(terminal, [SequenceSlice(0, ep.length - 1, 1)])
    values = torch.tensor([[0.0, float("nan")]])
    target, _, discount = nstep_targets(batch, values, 0.9, 1)
    assert torch.isfinite(target).all() and not discount.any()


@pytest.mark.parametrize("algorithm", ["qmix", "qplex", "qtran"])
@pytest.mark.parametrize("recurrent", [False, True])
@pytest.mark.parametrize("strategy", ["zero_burnin", "prefix_recompute"])
def test_each_algorithm_updates_all_required_models(algorithm, recurrent, strategy):
    learner = make_value_learner(algorithm, recurrent=recurrent, state_strategy=strategy)
    episodes = collect_values(learner)
    sources = [s for i, ep in episodes.items() for s in partition_episode(i, ep, length=2)]
    actor = deepcopy(learner.actor.state_dict())
    critic = deepcopy(learner.critic.state_dict())
    target = deepcopy(learner.target_actor.state_dict())
    metrics = learner.update(episodes, sources)
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert metrics["team_steps"] == 8
    assert any(not torch.equal(v, actor[k]) for k, v in learner.actor.state_dict().items())
    assert any(not torch.equal(v, critic[k]) for k, v in learner.critic.state_dict().items())
    assert_tree_equal(target, learner.target_actor.state_dict())
    assert all(p.grad is None and not p.requires_grad for p in learner.target_actor.parameters())
    if algorithm == "qplex":
        assert metrics["attention_regularization"] > 0
    if algorithm == "qtran":
        assert learner.objective_updates == 1 and "optimality_loss" in metrics
    learner.update(episodes, sources)
    assert_tree_equal(learner.actor.state_dict(), learner.target_actor.state_dict())
    assert_tree_equal(learner.critic.state_dict(), learner.target_critic.state_dict())
    snapshot = deepcopy(learner.state_dict())
    restored = make_value_learner(algorithm, recurrent=recurrent, state_strategy=strategy)
    restored.load_state_dict(snapshot)
    expected_metrics = learner.update(episodes, sources)
    actual_metrics = restored.update(episodes, sources)
    assert actual_metrics == expected_metrics
    assert_tree_equal(learner.state_dict(), restored.state_dict())


def test_qmix_monotonicity_and_masked_units():
    learner = make_value_learner("qmix", recurrent=False)
    mixing = learner.critic.mixing
    q = torch.randn(3, 2, requires_grad=True)
    state = torch.randn(3, 8)
    mixing(q, state).sum().backward()
    assert (q.grad >= 0).all()
    increased = q.detach() + torch.tensor([1.0, 0.0])
    assert (mixing(increased, state) >= mixing(q.detach(), state)).all()


def test_qplex_decomposition_igm_and_detached_advantage():
    learner = make_value_learner("qplex", recurrent=False)
    episodes = collect_values(learner, death=False)
    batch = collate_sequences(episodes, [SequenceSlice(0, 0, 2)])
    b, t, n = batch.frames["alive"].shape
    q = (
        torch.tensor([[[[3.0, 1.0, 90.0], [1.0, 4.0, 90.0]]]])
        .expand(b, t, -1, -1)
        .clone()
        .requires_grad_()
    )
    policy = PolicyOutput(q, torch.zeros(b, t, n, 8), {}, "q_values")
    alive, available = batch.frames["alive"], batch.frames["available_actions"]
    greedy = torch.tensor([[[0, 1]]]).expand(b, t, -1)
    state = learner.critic.initial_state(b, "cpu", torch.float32)
    valid = alive.any(-1)
    reset = torch.zeros_like(valid)

    def mix(actions):
        return learner.critic.forward_sequence(
            policy, batch.frames["states"], actions, alive, available, state, valid, reset
        )

    best = mix(greedy)
    assert not best.diagnostics["advantage"].any()
    torch.testing.assert_close(
        best.values, best.diagnostics["value"] + best.diagnostics["advantage"]
    )
    for a in (0, 1):
        for c in (0, 1):
            other = mix(torch.tensor([[[a, c]]]).expand(b, t, -1))
            assert (best.values >= other.values - 1e-6).all()
    assert (
        not best.diagnostics["advantage"].requires_grad
        or torch.autograd.grad(
            best.diagnostics["advantage"].sum(), q, allow_unused=True, retain_graph=True
        )[0]
        is None
    )


def test_counterfactual_excludes_own_action_and_dead_observation():
    learner = make_value_learner("qtran", recurrent=False)
    core = learner.critic.core
    features = torch.randn(4, 2, 8)
    actions = torch.zeros(4, 2, dtype=torch.long)
    before = core(features, actions)["q_per_action"]
    changed = actions.clone()
    changed[:, 0] = 1
    after = core(features, changed)["q_per_action"]
    torch.testing.assert_close(before[:, 0], after[:, 0])
    live = torch.tensor([[True, False]]).expand(4, -1)
    x = features.clone()
    x[:, 1] += 1000
    torch.testing.assert_close(
        core(features, actions, live)["q_per_action"], core(x, actions, live)["q_per_action"]
    )


def test_replay_eviction_sampling_and_roundtrip():
    learner = make_value_learner("qmix")
    episodes = collect_values(learner)
    replay = EpisodeReplay(2, "reward_weighted")
    assert replay.add(episodes[0]) == 0
    replay.add(episodes[1])
    replay.add(episodes[0])
    assert list(replay.episodes) == [1, 2]
    a = replay.sample(5, 2, torch.Generator().manual_seed(12))
    restored = EpisodeReplay(2, "reward_weighted")
    restored.load_state_dict(replay.state_dict())
    assert a == restored.sample(5, 2, torch.Generator().manual_seed(12))
    assert {s.episode_id for s in a}.issubset({1, 2})


def make_value_trainer(algorithm, workdir=None, **kwargs):
    from marlite.training.value import SequenceValueTrainer

    learner = make_value_learner(algorithm)
    return SequenceValueTrainer(
        learner,
        ParallelCollector(lambda: TinyParallel(mask=True, death=True), max_steps=4),
        EpisodeReplay(capacity=3),
        workdir=workdir,
        sequence_length=2,
        batch_sequences=3,
        updates_per_iteration=2,
        epsilon_decay_steps=20,
        **kwargs,
    )


@pytest.mark.parametrize("algorithm", ["qmix", "qplex", "qtran"])
def test_value_training_lifecycle_checkpoint_and_validation_protocol(algorithm, tmp_path):
    original = make_value_trainer(algorithm, tmp_path / "original")
    original.run(1, train_episodes=2, eval_episodes=2, checkpoint_interval=1)
    assert len(original.replay.episodes) == 2
    assert original.rollback_weights is None
    original.evaluate(2, purpose="test")
    assert len(original.replay.episodes) == 2
    assert original.test_round == 1 and original.evaluation_round == 1
    path = tmp_path / "resume.pt"
    original.save_checkpoint(path)
    expected_metrics = original.train_iteration(2)
    expected = deepcopy(original.learner.state_dict())
    expected_replay = deepcopy(original.replay.state_dict())
    restored = make_value_trainer(algorithm, tmp_path / "restored")
    restored.load_checkpoint(path)
    actual_metrics = restored.train_iteration(2)
    assert actual_metrics == expected_metrics
    assert_tree_equal(expected, restored.learner.state_dict())
    assert_tree_equal(expected_replay, restored.replay.state_dict())
    assert original.env_steps == restored.env_steps
    assert len(restored.replay.episodes) == 3
    assert actual_metrics["epsilon"] < 1.0


def test_opt_in_rollback_restores_weights_without_undoing_optimizer_steps():
    trainer = make_value_trainer("qtran", rollback_interval=1, validation_replay_ratio=0.5)
    trainer.train_iteration(2)
    trainer.evaluate(2)
    assert len(trainer.replay.episodes) == 3
    assert trainer.rollback_weights is not None
    optimizer_before = deepcopy(trainer.learner.actor_optimizer.state_dict())
    with torch.no_grad():
        for parameter in trainer.learner.actor.parameters():
            parameter.add_(100)
    metrics = trainer.train_iteration(2)
    assert metrics["rollback_applied"]
    assert max(float(p.detach().abs().max()) for p in trainer.learner.actor.parameters()) < 10
    step_before = next(iter(optimizer_before["state"].values()))["step"]
    step_after = next(iter(trainer.learner.actor_optimizer.state.values()))["step"]
    assert step_after == step_before + trainer.updates_per_iteration
    replay_before = trainer.replay.next_id
    result = trainer.evaluate(2, purpose="test")
    assert result["purpose"] == "test" and trainer.replay.next_id == replay_before


def test_ema_target_refresh_and_disabled_qtran_constraints():
    learner = make_value_learner("qtran", target_mode="ema", target_interval=1)
    learner.objective.lambda_opt = learner.objective.lambda_nopt = 0.0
    episodes = collect_values(learner)
    sources = partition_episode(0, episodes[0], 2)
    before_target = deepcopy(learner.target_actor.state_dict())
    before_value = deepcopy(learner.objective.state_dict())
    learner.update(episodes, sources)
    assert learner.objective_updates == 0 and not learner.objective_optimizer.state
    assert_tree_equal(before_value, learner.objective.state_dict())
    for key, value in learner.target_actor.state_dict().items():
        expected = torch.lerp(
            before_target[key], learner.actor.state_dict()[key], learner.target_tau
        )
        torch.testing.assert_close(value, expected)


def test_qtran_constraint_weights_match_a_hand_computed_counterexample():
    from types import SimpleNamespace
    from marlite.algorithm.sequence_mixers import JointOutput

    class ConstantJoint(nn.Module):
        def __init__(self):
            super().__init__()
            self.constant = nn.Parameter(torch.tensor(3.0))

        def forward(self, features, actions, alive_mask=None):
            return {"q_per_action": self.constant.expand(*actions.shape, 3)}

    learner = make_value_learner("qmix")
    episodes = collect_values(learner, death=False, max_steps=1)
    episodes[0].transitions["actions"][0] = torch.tensor([1, 0])
    batch = collate_sequences(episodes, [SequenceSlice(0, 0, 1)])
    q = torch.tensor(
        [[[[1.0, 0.0, 99.0], [0.0, 2.0, 99.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]],
        requires_grad=True,
    )
    policy = PolicyOutput(q, torch.zeros(1, 2, 2, 8), {}, "q_values")
    cf = torch.full((1, 2, 2, 3), 3.0, requires_grad=True)
    joint = JointOutput(torch.tensor([[3.0, 0.0]]), {}, {"counterfactual": cf})
    value = torch.ones(1, 1, requires_grad=True)
    core = ConstantJoint()
    objective = QTRANObjective(nn.Identity(), lambda_opt=0.3, lambda_nopt=0.7)
    terms = objective.terms(
        policy, joint, SimpleNamespace(core=core), batch, torch.zeros(1, 1), value
    )
    # Feasible greedy utilities sum to 3; joint=3 and V=1 -> optimal error=1.
    # Sampled utilities are both zero; each counterfactual minimum is -2.
    assert terms["optimality_loss"].total == 1
    assert terms["nonoptimality_loss"].total / terms["nonoptimality_loss"].count == 4
    total = sum(term.weight * term.total / term.count.clamp_min(1) for term in terms.values())
    assert float(total.detach()) == pytest.approx(12.1)
    total.backward()
    assert cf.grad is None and core.constant.grad is None
    torch.testing.assert_close(value.grad, torch.tensor([[-2.2]]))
    torch.testing.assert_close(q.grad[0, 0, :, 2], torch.zeros(2))  # Illegal large Q never wins.


@pytest.mark.parametrize("algorithm", ["qmix", "qplex", "qtran"])
def test_value_algorithms_train_on_real_mpe2(algorithm, tmp_path):
    import numpy as np
    from marlite.training.value import SequenceValueTrainer

    pytest.importorskip("mpe2")
    from mpe2 import simple_spread_v3

    def factory():
        return simple_spread_v3.parallel_env(N=2, max_cycles=3, continuous_actions=False)

    env = factory()
    try:
        observations, _ = env.reset(seed=1)
        obs_dim = int(np.prod(next(iter(observations.values())).shape))
        state_dim = int(np.prod(env.state().shape))
        action_dim = env.action_space(env.possible_agents[0]).n
    finally:
        env.close()
    learner = make_value_learner(
        algorithm, obs_dim=obs_dim, state_dim=state_dim, action_dim=action_dim, n_steps=3
    )
    trainer = SequenceValueTrainer(
        learner,
        ParallelCollector(factory, max_steps=3),
        EpisodeReplay(4),
        sequence_length=2,
        batch_sequences=3,
        updates_per_iteration=2,
        workdir=tmp_path,
    )
    trainer.run(1, train_episodes=2, eval_episodes=2, checkpoint_interval=1)
    assert trainer.env_steps == 6 and learner.actor_updates == 2
    assert len(trainer.replay.episodes) == 2
    assert (tmp_path / "checkpoints" / "iteration_1.pt").exists()


def test_inactive_replay_does_not_step_or_refresh_targets():
    learner = make_value_learner("qmix", target_interval=1)
    episodes = collect_values(learner, max_steps=1)
    ep = episodes[0]
    ep.frames["alive"].zero_()
    ep.frames["agent_reset"].zero_()
    ep.transitions["rewards"].zero_()
    before = deepcopy(learner.state_dict())
    metrics = learner.update({0: ep}, [SequenceSlice(0, 0, 1)])
    assert metrics["team_steps"] == 0
    assert_tree_equal(before, learner.state_dict())


def test_optimizer_ownership_is_checked():
    from marlite.learning.sequence import validate_optimizer_ownership

    a, b = nn.Linear(3, 4), nn.Linear(3, 4)
    wrong = torch.optim.Adam(b.parameters())
    with pytest.raises(ValueError, match="exactly"):
        validate_optimizer_ownership(((a, wrong),))
