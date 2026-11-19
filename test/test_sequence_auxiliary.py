"""Representation losses preserve masks, ownership and distributed objectives."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from marlite.algorithm.sequence_agent_group import PolicyOutput
from marlite.algorithm.sequence_value import SequenceValue
from marlite.algorithm.temporal import RecurrentGRU
from marlite.data import collate_sequences, partition_episode, SequenceSlice
from marlite.collection.parallel import ParallelCollector
from marlite.learning.auxiliary import (
    AuxiliaryLearning,
    ReconstructionObjective,
    ObservationTargets,
    ConstructorTargets,
    Targets,
    reparameterization_noise,
)
from marlite.learning.mappo import MAPPOLearner, prepare_ppo_targets
from marlite.learning.value import ValueLearner
from marlite.util.loss_func import PITLoss
from marlite.util.loss_mixer import WeightedSumLoss, EMAGradNormLoss, PCGradLoss
from test.test_sequence_communication import communication_actor
from test.test_sequence_mappo import TinyParallel, make_trainer, assert_tree_equal
from test.test_sequence_value import make_value_learner


def task_mixer(kind):
    if kind == "pit":
        return PITLoss(2, weights=[1.0, 0.2], reduction="sum")
    constructor = {"weighted": WeightedSumLoss, "ema": EMAGradNormLoss, "pcgrad": PCGradLoss}[kind]
    return constructor(weights=[1.0, 0.2])


def auxiliary_learner(
    algorithm="mappo",
    kind="bayesian",
    mixer="weighted",
    *,
    start_update=0,
    detach_fusion=False,
    variational=True,
):
    actor, provider = communication_actor(
        kind, "logits" if algorithm == "mappo" else "q_values", detach_fusion=detach_fusion
    )
    source = "agent" if kind == "graph_prob" else "group"
    if algorithm == "mappo":
        torch.manual_seed(10)
        critic = SequenceValue(nn.Linear(3, 8), RecurrentGRU(8, 8), nn.Linear(8, 1))
        kwargs = {}
    else:
        base = make_value_learner(algorithm)
        critic = base.critic
        if algorithm == "qtran":
            # QTRAN consumes the action-head input, including communication.
            critic.core.phi = nn.Linear(12 + 3, 8)
            critic.core.psi = nn.Linear(12, 8)
        kwargs = {"objective_optimizer": base.objective_optimizer}
    objective = ReconstructionObjective(
        nn.Linear(4, 3),
        ObservationTargets(source),
        source=source,
        variational=variational and kind != "ae",
        group_kl=0.02 if source == "group" and kind != "ae" else 0.0,
        agent_kl=0.01 if kind != "ae" else 0.0,
    )
    auxiliary = AuxiliaryLearning(
        objective,
        torch.optim.Adam(objective.parameters(), lr=0.002),
        mixer=task_mixer(mixer),
        start_update=start_update,
    )
    optimizers = [
        torch.optim.Adam(actor.parameters(), lr=0.001),
        torch.optim.Adam(critic.parameters(), lr=0.001),
    ]
    if algorithm == "mappo":
        learner = MAPPOLearner(actor, critic, *optimizers, auxiliary=auxiliary, burn_in=1)
    else:
        learner = ValueLearner(
            actor,
            critic,
            base.objective,
            *optimizers,
            auxiliary=auxiliary,
            n_steps=2,
            burn_in=1,
            **kwargs,
        )
    return learner, provider


def auxiliary_data(learner, provider):
    collector = ParallelCollector(
        lambda: TinyParallel(death=True, mask=True), max_steps=4, context_provider=provider
    )
    critic = learner.critic if learner.actor.output_kind == "logits" else None
    episodes = {i: collector.collect_episode(learner.actor, critic, seed=19 + i) for i in range(2)}
    if critic is not None:
        episodes = prepare_ppo_targets(episodes)
    sources = [s for i, ep in episodes.items() for s in partition_episode(i, ep, length=2)]
    return episodes, sources


@pytest.mark.parametrize("algorithm", ["mappo", "qmix", "qplex", "qtran"])
@pytest.mark.parametrize("mixer", ["weighted", "pit", "ema", "pcgrad"])
def test_auxiliary_updates_and_complete_resume(algorithm, mixer):
    learner, provider = auxiliary_learner(algorithm, mixer=mixer)
    episodes, sources = auxiliary_data(learner, provider)
    first = learner.update(episodes, sources)
    assert first["ssl/reconstruction_loss_count"] == 8
    assert all(torch.isfinite(torch.tensor(v)) for v in first.values())
    assert any(
        p.grad is not None and p.grad.abs().sum() for p in learner.auxiliary.objective.parameters()
    )
    checkpoint = deepcopy(learner.state_dict())
    reference = learner.update(episodes, sources)
    restored, _ = auxiliary_learner(algorithm, mixer=mixer)
    restored.load_state_dict(checkpoint)
    actual = restored.update(episodes, sources)
    assert actual == reference
    assert_tree_equal(restored.state_dict(), learner.state_dict())
    if mixer == "pit":
        assert learner.auxiliary.mixer.step == 2
    if mixer == "ema":
        assert learner.auxiliary.mixer.initialized


@pytest.mark.parametrize("kind", ["ae", "graph_prob", "sample_mean", "ci"])
def test_ae_graph_vae_and_consensus_reconstruction(kind):
    learner, provider = auxiliary_learner(kind=kind)
    episodes, sources = auxiliary_data(learner, provider)
    metrics = learner.update(episodes, sources)
    assert metrics["ssl/reconstruction_loss"] >= 0
    if kind == "ae":
        assert "ssl/agent_kl" not in metrics
    else:
        assert metrics["ssl/agent_kl"] >= 0


def test_ssl_warmup_and_detached_rl_do_not_drift_inactive_heads():
    learner, provider = auxiliary_learner(start_update=1, detach_fusion=True)
    episodes, sources = auxiliary_data(learner, provider)
    head = deepcopy(learner.auxiliary.objective.state_dict())
    messages = deepcopy(learner.actor.message_encoder.state_dict())
    first = learner.update(episodes, sources)
    assert "ssl/reconstruction_loss" not in first
    assert_tree_equal(head, learner.auxiliary.objective.state_dict())
    assert_tree_equal(messages, learner.actor.message_encoder.state_dict())
    learner.update(episodes, sources)
    assert any(
        p.grad is not None and p.grad.abs().sum()
        for p in learner.actor.message_encoder.parameters()
    )
    assert learner.auxiliary.updates == 2


@pytest.mark.parametrize("weighting,expected", [("groups", 2.5), ("agents", 2.0)])
def test_reconstruction_population_and_mask_are_explicit(weighting, expected):
    class Target:
        def __call__(self, batch, diagnostics, length):
            return Targets(
                torch.tensor([[[[2.0], [6.0], [float("nan")]]]]),
                torch.tensor([[[True, True, False]]]),
            )

    mean = torch.tensor([[[[1.0], [4.0], [0.0]]]], requires_grad=True)
    diagnostics = {
        "group_mu": mean,
        "group_log_var": torch.zeros_like(mean),
        "group_valid": torch.tensor([[[True, True, False]]]),
        "groups": torch.tensor([[[0, 0, 1, -1]]]),
    }
    batch = SimpleNamespace(
        loss_mask=torch.ones(1, 1, dtype=torch.bool),
        frames={"alive": torch.tensor([[[True, True, True, False]]])},
    )
    output = PolicyOutput(torch.empty(0), torch.empty(0), {}, "logits", diagnostics)
    objective = ReconstructionObjective(
        nn.Identity(), Target(), group_kl=0.1, group_weighting=weighting
    )
    terms = objective(output, batch)
    term = terms["reconstruction_loss"]
    assert float((term.total / term.count).detach()) == pytest.approx(expected)
    assert float(terms["group_kl"].total.detach()) == pytest.approx(8.5)
    term.total.backward()
    assert not mean.grad[0, 0, 2].any()


def test_entity_target_mask_averages_feature_channels_only():
    class Target:
        def __call__(self, *args):
            return Targets(
                torch.tensor([[[[[1.0, 3.0], [999.0, 999.0]]]]]), torch.tensor([[[[True, False]]]])
            )

    mean = torch.zeros(1, 1, 1, 4, requires_grad=True)
    output = SimpleNamespace(
        diagnostics={"group_mu": mean, "group_valid": torch.ones(1, 1, 1, dtype=torch.bool)}
    )
    batch = SimpleNamespace(loss_mask=torch.ones(1, 1, dtype=torch.bool))
    objective = ReconstructionObjective(nn.Unflatten(-1, (2, 2)), Target())
    term = objective(output, batch)["reconstruction_loss"]
    assert term.total == 5 and term.count == 1
    term.total.backward()
    assert mean.grad.tolist() == [[[[-1.0, -3.0, 0.0, 0.0]]]]


def test_reparameterization_is_stable_under_partition_and_uses_fresh_draws():
    learner, provider = auxiliary_learner()
    episodes, sources = auxiliary_data(learner, provider)
    batch = collate_sequences(episodes, sources)
    ref = torch.zeros(4, 2, 3, 4)
    whole = reparameterization_noise(batch, ref, 3, 17)
    selected = collate_sequences(episodes, [sources[2], sources[0]])
    subset = reparameterization_noise(selected, ref[:2], 3, 17)
    torch.testing.assert_close(subset, whole[[2, 0]], rtol=0, atol=0)
    assert not torch.equal(whole, reparameterization_noise(batch, ref, 4, 17))


@pytest.mark.parametrize("topology", ["groups", "graph"])
def test_constructor_adapter_labels_each_frame_and_edge_direction(topology):
    class Constructor:
        def process(self, observations, states, topology_arg, alive):
            assert observations.shape[:2] == (4, 1)
            if topology == "graph":
                assert topology_arg[0][0].tolist() == [[0], [1]]
            else:
                assert topology_arg.shape == (4, 2)
            return states[:, 0, None, :], ~alive[:, 0, :1]

    learner, provider = auxiliary_learner()
    episodes, _ = auxiliary_data(learner, provider)
    batch = collate_sequences(episodes, [SequenceSlice(0, 0, 2), SequenceSlice(1, 0, 2)])
    adjacency = torch.zeros(2, 3, 2, 2)
    adjacency[..., 1, 0] = 1
    diagnostics = {"groups": batch.frames["context.groups"], "adjacency": adjacency}
    builder = ConstructorTargets(
        Constructor(), topology, mask_is_padding=True, graph_source="current"
    )
    target = builder(batch, diagnostics, 2)
    torch.testing.assert_close(target.values[:, :, 0], batch.frames["states"][:, :2])
    assert target.valid.all()


def test_auxiliary_ownership_and_checkpoint_composition_reject_mistakes():
    learner, _ = auxiliary_learner()
    other = make_trainer().learner
    with pytest.raises(ValueError, match="composition"):
        other.load_state_dict(learner.state_dict())
    auxiliary = AuxiliaryLearning(learner.actor, learner.actor_optimizer)
    with pytest.raises(ValueError, match="must not own"):
        auxiliary.initialize(learner.backend, (learner.actor,))


def test_training_logs_and_checkpoint_cross_auxiliary_warmup(tmp_path):
    import json
    from marlite.training.mappo import SequenceMAPPOTrainer

    def trainer_at(directory):
        learner, provider = auxiliary_learner(start_update=1, mixer="pit")
        collector = ParallelCollector(
            lambda: TinyParallel(death=True, mask=True), max_steps=4, context_provider=provider
        )
        return SequenceMAPPOTrainer(
            learner,
            collector,
            sequence_length=2,
            batch_sequences=2,
            ppo_epochs=2,
            workdir=directory,
        )

    trainer = trainer_at(tmp_path / "run")
    trainer.run(1, train_episodes=2, eval_episodes=2, checkpoint_interval=1)
    event = json.loads((tmp_path / "run" / "events.jsonl").read_text())
    assert event["training"]["ssl/reconstruction_loss_count"] == 12
    restored = trainer_at(tmp_path / "restored")
    restored.load_checkpoint(tmp_path / "run" / "checkpoints" / "iteration_1.pt")
    assert trainer.train_iteration(2) == restored.train_iteration(2)
    assert_tree_equal(trainer.learner.state_dict(), restored.learner.state_dict())


def test_actual_magent_group_semantic_constructor_preserves_time_and_liveness():
    from marlite.util.self_supervised_data_constructor.magent_vec_state_group_features_constructor import (
        MagentVecStateGroupFeaturesConstructor,
    )

    # Two own slots and one enemy; second own slot dies before frame two.
    states = torch.zeros(1, 3, 3, 5)
    states[..., 0] = 1.0
    states[:, :, 2, 1] = 1.0
    states[:, :, 0, 3:] = torch.tensor([0.2, 0.2])
    states[:, :, 1, 3:] = torch.tensor([0.3, 0.3])
    states[:, :, 2, 3:] = torch.tensor([0.25, 0.25])
    live = torch.tensor([[[True, True], [True, False], [True, False]]])
    groups = torch.tensor([[[0, 0], [0, -1], [0, -1]]])
    batch = SimpleNamespace(
        frames={"states": states, "alive": live, "observations": torch.zeros(1, 3, 2, 3)},
        filled=torch.ones(1, 2, dtype=torch.bool),
    )
    constructor = MagentVecStateGroupFeaturesConstructor(2, 0.2, n_offsets=0)
    target = ConstructorTargets(constructor)(batch, {"groups": groups}, 2)
    assert target.valid.tolist() == [[[True, False], [True, False]]]
    # Feature 3 is total own HP; feature 8 is deduplicated observed enemy HP.
    torch.testing.assert_close(
        target.values[0, :, 0, 3], torch.tensor([2.0, 1.0], dtype=torch.float16)
    )
    torch.testing.assert_close(
        target.values[0, :, 0, 8], torch.tensor([1.0, 1.0], dtype=torch.float16)
    )


@pytest.mark.parametrize("algorithm", ["mappo", "qmix"])
def test_communication_vae_trains_on_real_mpe2(algorithm, tmp_path):
    import numpy as np
    from marlite.algorithm.sequence_agent_group import SequenceAgentGroup
    from marlite.communication.fusion import GaussianConsensus
    from marlite.communication.context import AllTogether
    from marlite.training.mappo import SequenceMAPPOTrainer
    from marlite.training.value import SequenceValueTrainer
    from marlite.data.replay import EpisodeReplay

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
    actor = SequenceAgentGroup(
        nn.Linear(obs_dim, 8),
        RecurrentGRU(8, 8),
        nn.Linear(12, action_dim),
        "logits" if algorithm == "mappo" else "q_values",
        communication=GaussianConsensus(4, 2),
        message_encoder=nn.Linear(8, 8),
    )
    objective = ReconstructionObjective(
        nn.Linear(4, obs_dim),
        ObservationTargets("group"),
        variational=True,
        group_kl=0.01,
        agent_kl=0.01,
    )
    auxiliary = AuxiliaryLearning(
        objective, torch.optim.Adam(objective.parameters()), mixer=task_mixer("pcgrad")
    )
    if algorithm == "mappo":
        critic = SequenceValue(nn.Linear(state_dim, 8), RecurrentGRU(8, 8), nn.Linear(8, 1))
        learner = MAPPOLearner(
            actor,
            critic,
            torch.optim.Adam(actor.parameters()),
            torch.optim.Adam(critic.parameters()),
            auxiliary=auxiliary,
            burn_in=1,
        )
    else:
        base = make_value_learner(
            "qmix", obs_dim=obs_dim, state_dim=state_dim, action_dim=action_dim
        )
        critic = base.critic
        learner = ValueLearner(
            actor,
            critic,
            base.objective,
            torch.optim.Adam(actor.parameters()),
            torch.optim.Adam(critic.parameters()),
            auxiliary=auxiliary,
            burn_in=1,
        )
    collector = ParallelCollector(factory, max_steps=3, context_provider=AllTogether())
    if algorithm == "mappo":
        trainer = SequenceMAPPOTrainer(
            learner, collector, sequence_length=2, batch_sequences=2, ppo_epochs=1, workdir=tmp_path
        )
    else:
        trainer = SequenceValueTrainer(
            learner,
            collector,
            EpisodeReplay(4),
            sequence_length=2,
            batch_sequences=2,
            updates_per_iteration=2,
            workdir=tmp_path,
        )
    trainer.run(1, train_episodes=2, eval_episodes=2, checkpoint_interval=1)
    assert trainer.env_steps == 6 and auxiliary.updates == 2
    assert (tmp_path / "checkpoints" / "iteration_1.pt").exists()


def test_zero_ssl_weight_preserves_momentum_and_head_weights():
    learner, provider = auxiliary_learner()
    episodes, sources = auxiliary_data(learner, provider)
    learner.update(episodes, sources)
    head = deepcopy(learner.auxiliary.objective.state_dict())
    optimizer = deepcopy(learner.auxiliary.optimizer.state_dict())
    learner.auxiliary.mixer.weights[1] = 0
    learner.update(episodes, sources)
    assert all(p.grad is None for p in learner.auxiliary.objective.parameters())
    assert_tree_equal(head, learner.auxiliary.objective.state_dict())
    assert_tree_equal(optimizer, learner.auxiliary.optimizer.state_dict())


def test_custom_set_loss_excludes_padding_before_matching():
    from marlite.util.loss_func import ChamferDistanceLoss

    prediction = torch.tensor([[[0.0], [10.0]], [[2.0], [3.0]]], requires_grad=True)
    truth = torch.tensor([[[5.0], [0.0]], [[float("nan")], [float("nan")]]])
    mask = torch.tensor([[True, False], [False, False]])
    criterion = ChamferDistanceLoss(reduction="none")
    loss = criterion(prediction, truth, mask)
    torch.testing.assert_close(loss, torch.tensor([50.0, 0.0]))
    loss.sum().backward()
    assert prediction.grad.tolist() == [[[-20.0], [0.0]], [[0.0], [0.0]]]

    class Target:
        def __call__(self, *args):
            return Targets(truth[:1].reshape(1, 1, 1, 2, 1), mask[:1].reshape(1, 1, 1, 2))

    source = prediction[:1].detach().reshape(1, 1, 1, 2)
    output = SimpleNamespace(
        diagnostics={"group_mu": source, "group_valid": torch.ones(1, 1, 1, dtype=torch.bool)}
    )
    batch = SimpleNamespace(loss_mask=torch.ones(1, 1, dtype=torch.bool))
    objective = ReconstructionObjective(
        nn.Unflatten(-1, (2, 1)), Target(), reconstruction=criterion
    )
    term = objective(output, batch)["reconstruction_loss"]
    assert term.total == 50 and term.count == 1
