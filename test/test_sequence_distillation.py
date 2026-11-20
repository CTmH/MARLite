"""QMPD components preserve local-only encoding and frozen-teacher gradients."""

from copy import deepcopy

import pytest
import torch
from torch import nn

from marlite.algorithm.sequence_agent_group import SequenceAgentGroup, DecisionInput
from marlite.algorithm.sequence_value import SequenceValue
from marlite.algorithm.temporal import RecurrentGRU
from marlite.algorithm.model.group_communication import GroupSetFusion, ScalarMessageCodec
from marlite.collection.parallel import ParallelCollector
from marlite.communication.context import AllTogether, frame_context
from marlite.communication.group_set import GroupSetCommunication, QuantizedCommunication
from marlite.data import collate_sequences, partition_episode
from marlite.learning.auxiliary import AuxiliaryLearning
from marlite.learning.distillation import (
    FrozenPolicyReference,
    ActionDistillationObjective,
    CompositeObjective,
    FrameQueryTargets,
    RelationQueryObjective,
    codec_distillation,
    replacement_senders,
)
from marlite.algorithm.auxiliary.query_prediction import QueryPredictionHead
from marlite.learning.mappo import MAPPOLearner, prepare_ppo_targets
from marlite.util.communication_training import StateThresholdQueries
from test.test_sequence_mappo import TinyParallel, assert_tree_equal


def distillation_components(
    *, online_weight=0.2, relation=False, device="cpu", obs_dim=3, state_dim=3, action_dim=3
):
    torch.manual_seed(63)
    aggregation = GroupSetCommunication(
        GroupSetFusion(8, 2, 3, 4),
        nn.MultiheadAttention(8, 2, dropout=0.0, batch_first=True),
        header_bytes=9,
    )
    teacher = SequenceAgentGroup(
        nn.Linear(obs_dim, 8),
        RecurrentGRU(8, 8),
        nn.Linear(8, action_dim),
        communication=aggregation,
        readout=nn.Sequential(nn.Linear(16, 8), nn.GELU()),
    )
    student = deepcopy(teacher)
    student.communication = QuantizedCommunication(
        ScalarMessageCodec(8, 4, 4), student.communication, header_bytes=9
    )
    teacher.to(device)
    reference = FrozenPolicyReference(teacher, capacity=2)
    objective = {
        "distillation": ActionDistillationObjective(reference, online_weight, decay_updates=3)
    }
    if relation:
        queries = StateThresholdQueries([0, 1], [[1.0, 0.0], [0.0, 1.0]], [0.5, 0.5])
        objective["relation"] = RelationQueryObjective(
            QueryPredictionHead(nn.Linear(10, 1)), FrameQueryTargets(queries), weight=0.1
        )
    objectives = CompositeObjective(objective)
    parameters = list(objectives.parameters())
    auxiliary = AuxiliaryLearning(
        objectives, torch.optim.Adam(parameters, lr=0.001) if parameters else None
    )
    critic = SequenceValue(nn.Linear(state_dim, 8), RecurrentGRU(8, 8), nn.Linear(8, 1))
    learner = MAPPOLearner(
        student,
        critic,
        torch.optim.Adam(student.parameters(), lr=0.001),
        torch.optim.Adam(critic.parameters(), lr=0.001),
        auxiliary=auxiliary,
        burn_in=1,
    )
    return learner, reference


def distillation_data(learner, *, raw=False, reference=None):
    actor = reference.actor if raw else learner.actor
    collector = ParallelCollector(
        lambda: TinyParallel(death=True, mask=True), max_steps=4, context_provider=AllTogether()
    )
    episodes = {i: collector.collect_episode(actor, learner.critic, seed=90 + i) for i in range(2)}
    prepared = prepare_ppo_targets(episodes)
    sources = [
        source for key, ep in episodes.items() for source in partition_episode(key, ep, length=2)
    ]
    return prepared, sources


@pytest.mark.parametrize("bits,width", [(1, 8), (2, 4), (4, 4), (8, 4)])
def test_hard_codec_matches_transmitted_integer_decode_exactly(bits, width):
    torch.manual_seed(7)
    codec = ScalarMessageCodec(8, width, bits)
    x = torch.randn(100, 8, requires_grad=True)
    decoded, codes = codec(x)
    assert torch.equal(decoded, codec.decode(codes))
    assert torch.equal(codes, codec.unpack(codec.pack(codes)))
    decoded.square().sum().backward()
    assert x.grad.abs().sum() > 0 and codec.encoder.weight.grad.abs().sum() > 0


def test_group_attention_is_permutation_equivariant_and_records_actual_codes():
    learner, _ = distillation_components()
    actor = learner.actor
    obs = torch.randn(1, 4, 3, 3)
    live = torch.ones(1, 4, 3, dtype=torch.bool)
    reset = torch.zeros_like(live)
    groups = torch.tensor([[[0, 0, 1]]]).expand_as(live)
    result = actor(
        DecisionInput(obs, live, reset, {"groups": groups}),
        actor.initial_state(1, 3, "cpu", torch.float32),
    )
    order = torch.tensor([2, 0, 1])
    reordered = actor(
        DecisionInput(
            obs[:, :, order], live[:, :, order], reset[:, :, order], {"groups": groups[:, :, order]}
        ),
        actor.initial_state(1, 3, "cpu", torch.float32),
    )
    torch.testing.assert_close(result.values[:, :, order], reordered.values)
    assert result.diagnostics["record.communication_bytes"].unique().tolist() == [11]
    messages = result.diagnostics["message_codes"]
    original = actor.communication.codec.decode(messages)
    torch.testing.assert_close(original, result.diagnostics["decoded_messages"], rtol=0, atol=0)
    episodes, sources = distillation_data(learner)
    assert episodes[0].transitions["behavior.message_codes"].dtype == torch.uint8
    _, _, metrics = learner.losses(episodes, sources)
    assert abs(metrics["approx_kl"]) < 1e-6


def test_codec_distillation_keeps_teacher_frozen_but_transmits_message_gradients():
    learner, reference = distillation_components()
    episodes, sources = distillation_data(learner, raw=True, reference=reference)
    batch = collate_sequences(episodes, sources)
    live = batch.frames["alive"][:, :-1]
    weights = deepcopy(reference.actor.state_dict())
    codec = learner.actor.communication.codec
    predicted, term = codec_distillation(reference, codec, batch, episodes, live)
    (term.total / term.count).backward()
    assert codec.encoder.weight.grad is not None and codec.encoder.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in reference.actor.parameters())
    assert_tree_equal(weights, reference.actor.state_dict())
    assert predicted.diagnostics["relation_context"].requires_grad
    _, same = codec_distillation(reference, codec, batch, episodes, torch.zeros_like(live))
    assert abs(float(same.total.detach())) < 1e-6
    assert len(reference.cache) <= 2


@pytest.mark.parametrize("relation", [False, True])
def test_online_optional_distillation_and_query_head_resume(relation):
    learner, reference = distillation_components(relation=relation)
    episodes, sources = distillation_data(learner)
    first = learner.update(episodes, sources)
    assert first["ssl/online_distillation_loss"] > 0
    if relation:
        assert first["ssl/relation_prediction_loss_count"] == 16
    checkpoint = deepcopy(learner.state_dict())
    next_metrics = learner.update(episodes, sources)
    restored, _ = distillation_components(relation=relation)
    restored.load_state_dict(checkpoint)
    assert restored.update(episodes, sources) == next_metrics
    assert_tree_equal(learner.state_dict(), restored.state_dict())
    learner.update(episodes, sources)
    exhausted = learner.update(episodes, sources)
    assert "ssl/online_distillation_loss" not in exhausted
    assert all(p.grad is None for p in reference.actor.parameters())


def test_queries_and_replacements_do_not_depend_on_rank_or_rng_state():
    learner, _ = distillation_components(relation=True)
    episodes, sources = distillation_data(learner)
    batch = collate_sequences(episodes, sources)
    actor = learner.actor
    output = actor(
        DecisionInput(
            batch.frames["observations"][:, :-1],
            batch.frames["alive"][:, :-1],
            batch.frames["agent_reset"][:, :-1],
            frame_context(batch.frames, stop=-1),
        ),
        actor.initial_state(4, 2, "cpu", torch.float32),
    )
    objective = learner.auxiliary.objective.objectives["relation"]
    before = torch.get_rng_state().clone()
    queries = objective.targets(batch, output.diagnostics, 2, 3)
    assert torch.equal(before, torch.get_rng_state())
    groups = batch.frames["context.groups"][:, :-1]
    live = batch.frames["alive"][:, :-1]
    selected = replacement_senders(batch, groups, live, mode="progressive", stage=1, update=3)
    subset = collate_sequences(episodes, [sources[2], sources[0]])
    chosen = replacement_senders(
        subset, groups[[2, 0]], live[[2, 0]], mode="progressive", stage=1, update=3
    )
    assert torch.equal(chosen, selected[[2, 0]])
    diagnostics = {key: value[[2, 0]] for key, value in output.diagnostics.items()}
    other = objective.targets(subset, diagnostics, 2, 3)
    for key in queries:
        torch.testing.assert_close(other[key], queries[key][[2, 0]], rtol=0, atol=0)


def distillation_trainer(
    workdir=None, *, relation=False, replacement="all", offline_enabled=True, online_weight=0.2
):
    from marlite.training.distillation import SequenceDistillationTrainer

    learner, reference = distillation_components(relation=relation, online_weight=online_weight)
    student_relation = probe = None
    if relation:
        student_relation = learner.auxiliary.objective.objectives["relation"]
        probe = deepcopy(student_relation)
    collector = ParallelCollector(
        lambda: TinyParallel(death=True, mask=True), max_steps=4, context_provider=AllTogether()
    )
    return SequenceDistillationTrainer(
        learner,
        collector,
        reference,
        deepcopy(learner.critic),
        offline_enabled=offline_enabled,
        train_episodes=2,
        validation_episodes=2,
        codec_updates=3,
        policy_updates=2,
        probe_objective=probe,
        probe_updates=1 if relation else 0,
        student_relation=student_relation,
        codec_aux_weight=0.1 if relation else 0.0,
        policy_aux_weight=0.1 if relation else 0.0,
        replacement=replacement,
        replacement_steps=[1, 1, 1] if replacement == "progressive" else None,
        offline_batch_sequences=3,
        sequence_length=2,
        batch_sequences=3,
        ppo_epochs=1,
        workdir=workdir,
    )


@pytest.mark.parametrize("relation,replacement", [(False, "all"), (True, "progressive")])
def test_complete_qmpd_stages_and_offline_boundary_resume(tmp_path, relation, replacement):
    trainer = distillation_trainer(tmp_path / "run", relation=relation, replacement=replacement)
    teacher = deepcopy(trainer.reference.actor.state_dict())
    local = deepcopy(trainer.learner.actor.frame_encoder.state_dict())
    action = deepcopy(trainer.learner.actor.action_head.state_dict())
    # With a probe, pause just after probe fitting; otherwise after codec step 1.
    trainer.run_offline(max_updates=1, checkpoint_interval=1)
    assert_tree_equal(local, trainer.learner.actor.frame_encoder.state_dict())
    assert_tree_equal(action, trainer.learner.actor.action_head.state_dict())
    assert trainer.learner.actor_updates == 0 and trainer.learner.auxiliary.updates == 0
    assert not {ep.seed for ep in trainer.offline_train.values()} & {
        ep.seed for ep in trainer.offline_validation.values()
    }
    restored = distillation_trainer(
        tmp_path / "restored", relation=relation, replacement=replacement
    )
    restored.load_checkpoint(tmp_path / "run" / "checkpoints" / "offline_1.pt")
    trainer.run_offline()
    restored.run_offline()
    assert trainer.phase == restored.phase == "online"
    assert trainer.offline_counters == dict(probe=1 if relation else 0, codec=3, policy=2)
    assert trainer.offline_metrics == restored.offline_metrics
    assert_tree_equal(trainer.learner.state_dict(), restored.learner.state_dict())
    for name in trainer.stage_optimizers:
        optimizer = trainer.stage_optimizers[name]
        if optimizer is not None:
            assert_tree_equal(optimizer.state_dict(), restored.stage_optimizers[name].state_dict())
    assert_tree_equal(local, trainer.learner.actor.frame_encoder.state_dict())
    assert_tree_equal(teacher, trainer.reference.actor.state_dict())
    if relation:
        assert not any(p.requires_grad for p in trainer.probe_objective.parameters())
    first = trainer.train_iteration(2)
    second = restored.train_iteration(2)
    assert first == second
    assert_tree_equal(trainer.learner.state_dict(), restored.learner.state_dict())
    trainer.save_checkpoint(tmp_path / "online.pt")
    resumed = distillation_trainer(
        tmp_path / "online_restore", relation=relation, replacement=replacement
    )
    resumed.load_checkpoint(tmp_path / "online.pt")
    assert resumed.phase == "online"
    assert trainer.train_iteration(2) == resumed.train_iteration(2)
    assert_tree_equal(trainer.learner.state_dict(), resumed.learner.state_dict())


def test_qmpd_offline_disabled_collects_no_teacher_training_data(tmp_path):
    trainer = distillation_trainer(tmp_path, offline_enabled=False)
    trainer.run(1, train_episodes=2, eval_episodes=2, checkpoint_interval=1)
    assert trainer.phase == "online" and not trainer.offline_train
    assert not any(trainer.offline_counters.values())
    assert not (tmp_path / "offline_events.jsonl").exists()
    assert trainer.learner.actor_updates == 2


def test_attention_shared_key_bias_is_an_unidentifiable_direction():
    torch.manual_seed(12)
    attention = nn.MultiheadAttention(8, 2, dropout=0.0, batch_first=True)
    query, keys = torch.randn(2, 3, 8), torch.randn(2, 4, 8)
    original = attention(query, keys, keys, need_weights=False)[0]
    with torch.no_grad():
        attention.in_proj_bias[8:16] += torch.randn(8)
    shifted = attention(query, keys, keys, need_weights=False)[0]
    torch.testing.assert_close(original, shifted, rtol=1e-5, atol=1e-6)
    shifted.square().sum().backward()
    assert attention.in_proj_bias.grad[8:16].abs().max() < 1e-6


@torch.no_grad()
def distillation_probe(actor):
    observations = torch.arange(18, dtype=torch.float32).reshape(1, 3, 2, 3) / 18
    alive = torch.ones(1, 3, 2, dtype=torch.bool)
    reset = torch.zeros_like(alive)
    reset[:, 0] = True
    groups = torch.zeros_like(alive, dtype=torch.long)
    return actor(
        DecisionInput(observations, alive, reset, {"groups": groups}),
        actor.initial_state(1, 2, "cpu", torch.float32),
    ).values


def test_received_code_interventions_replace_only_selected_senders():
    learner, _ = distillation_components()
    actor = learner.actor
    obs = torch.randn(1, 2, 3)
    alive = torch.ones(1, 2, dtype=torch.bool)
    reset = torch.zeros_like(alive)
    groups = torch.zeros_like(alive, dtype=torch.long)
    initial = actor.initial_state(1, 2, "cpu", torch.float32)
    actual = actor.step(DecisionInput(obs, alive, reset, {"groups": groups}), initial)
    received = actual.diagnostics["message_codes"].clone()
    replayed = actor.step(
        DecisionInput(obs, alive, reset, {"groups": groups, "received_codes": received}), initial
    )
    torch.testing.assert_close(replayed.values, actual.values, rtol=0, atol=0)
    received[:, 0] = (received[:, 0] + 3) % 16
    received[:, 1] = 255  # Unselected sender padding is deliberately not a packet.
    changed = actor.step(
        DecisionInput(
            obs,
            alive,
            reset,
            {
                "groups": groups,
                "received_codes": received,
                "replace_senders": torch.tensor([[True, False]]),
            },
        ),
        initial,
    )
    assert torch.equal(changed.diagnostics["message_codes"][:, 0], received[:, 0])
    assert torch.equal(
        changed.diagnostics["message_codes"][:, 1], actual.diagnostics["message_codes"][:, 1]
    )
    assert (changed.values - actual.values).abs().max() > 1e-6
    torch.testing.assert_close(
        changed.diagnostics["local_features"], actual.diagnostics["local_features"]
    )


def test_qmpd_complete_short_workflow_on_real_mpe2(tmp_path):
    import numpy as np
    from marlite.training.distillation import SequenceDistillationTrainer

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
    learner, reference = distillation_components(
        obs_dim=obs_dim, state_dim=state_dim, action_dim=action_dim, relation=True
    )
    relation = learner.auxiliary.objective.objectives["relation"]
    collector = ParallelCollector(factory, max_steps=3, context_provider=AllTogether())
    trainer = SequenceDistillationTrainer(
        learner,
        collector,
        reference,
        deepcopy(learner.critic),
        train_episodes=2,
        validation_episodes=1,
        codec_updates=2,
        policy_updates=1,
        probe_objective=deepcopy(relation),
        probe_updates=1,
        student_relation=relation,
        codec_aux_weight=0.1,
        policy_aux_weight=0.1,
        offline_batch_sequences=2,
        sequence_length=2,
        batch_sequences=2,
        ppo_epochs=1,
        workdir=tmp_path,
    )
    trainer.run(1, train_episodes=2, eval_episodes=2, checkpoint_interval=1)
    assert trainer.phase == "online" and trainer.env_steps == 6
    assert trainer.offline_counters == dict(probe=1, codec=2, policy=1)
    assert trainer.learner.actor_updates == 2
    assert (tmp_path / "checkpoints" / "iteration_1.pt").exists()
