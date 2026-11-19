"""Communication is time-aligned and algorithm-independent, including its gradients."""

from copy import deepcopy

import pytest
import torch
from torch import nn

from marlite.algorithm.sequence_agent_group import DecisionInput, SequenceAgentGroup
from marlite.algorithm.temporal import RecurrentGRU
from marlite.algorithm.model.g2anet_attention import G2ANetAttention
from marlite.communication.context import AllTogether, FixedGraph, FixedGroups, StateBuilderContext, frame_context
from marlite.communication.fusion import GaussianConsensus, SetCommunication, MaskedMeanPool
from marlite.communication.graph import GraphCommunication, DenseGraphModel
from marlite.collection.parallel import ParallelCollector
from marlite.data import partition_episode, collate_sequences
from marlite.learning.mappo import MAPPOLearner, prepare_ppo_targets
from marlite.learning.value import ValueLearner
from test.test_sequence_mappo import make_trainer, TinyParallel
from test.test_sequence_value import make_value_learner


KINDS = ["graph", "g2anet", "graph_prob", "set", "set_prob", "ae", "bayesian", "ci", "sample_mean", "sampled"]


def communication_actor(kind, output_kind="logits", *, source="history", detach_messages=False, detach_fusion=False):
    torch.manual_seed(31)
    provider = None
    latent = 4
    if kind in ("graph", "g2anet", "graph_prob"):
        width = 8 if kind != "graph_prob" else 2 * latent
        graph = DenseGraphModel([nn.Sequential(nn.Linear(8, 8), nn.GELU())], nn.Linear(8, width))
        learned = G2ANetAttention(2, True, 8, 8) if kind == "g2anet" else None
        comm = GraphCommunication(graph, learned_graph=learned, normalize=kind != "g2anet",
                                  probabilistic=kind == "graph_prob", latent_dim=latent)
        if kind != "g2anet":
            provider = FixedGraph(2, [(0, 1), (1, 0)], self_loops=True)
        width = latent if kind == "graph_prob" else 8
        message = nn.Linear(3 if source == "observation" else 8, 8)
    elif kind in ("set", "set_prob"):
        comm = SetCommunication(MaskedMeanPool(), nn.Linear(8, 8 if kind == "set_prob" else 4),
                                 latent_dim=4, probabilistic=kind == "set_prob")
        message, width = nn.Linear(3 if source == "observation" else 8, 8), 4
    else:
        mode = "bayesian" if kind == "sampled" else kind
        comm = GaussianConsensus(4, 3, mode, sampled_policy=kind == "sampled")
        message = nn.Linear(3 if source == "observation" else 8, 4 if mode == "ae" else 8)
        provider, width = AllTogether(), 4
    return SequenceAgentGroup(nn.Linear(3, 8), RecurrentGRU(8, 8), nn.Linear(8 + width, 3),
                               output_kind, communication=comm, message_encoder=message,
                               message_source=source, detach_messages=detach_messages,
                               detach_fusion=detach_fusion), provider


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("algorithm", ["mappo", "qmix"])
def test_communication_collect_replay_and_update(kind, algorithm):
    actor, provider = communication_actor(kind, "logits" if algorithm == "mappo" else "q_values")
    collector = ParallelCollector(lambda: TinyParallel(death=True, mask=True), max_steps=4,
                                   context_provider=provider)
    if algorithm == "mappo":
        base = make_trainer()
        critic = base.learner.critic
        learner = MAPPOLearner(actor, critic, torch.optim.Adam(actor.parameters(), lr=.001),
                               torch.optim.Adam(critic.parameters(), lr=.001), burn_in=2)
        ep = collector.collect_episode(actor, critic, seed=19)
        episodes = prepare_ppo_targets({0: ep})
    else:
        base = make_value_learner("qmix")
        critic = base.critic
        learner = ValueLearner(actor, critic, base.objective, torch.optim.Adam(actor.parameters(), lr=.001),
                               torch.optim.Adam(critic.parameters(), lr=.001), n_steps=3, burn_in=2)
        ep = collector.collect_episode(actor, seed=19, epsilon=.3)
        episodes = {0: ep}
    sources = partition_episode(0, ep, length=2)
    if algorithm == "mappo":
        _, _, metrics = learner.losses(episodes, sources)
        assert abs(metrics["approx_kl"]) < 1e-6
    result = learner.update(episodes, sources)
    assert all(torch.isfinite(torch.tensor(v)) for v in result.values())
    if kind == "g2anet":
        hard = actor.communication.learned_graph.hard_attention_fc.weight
        assert hard.grad is not None and hard.grad.abs().sum() > 0
    if provider is not None:
        expected = "context.adjacency" if kind.startswith("graph") else "context.groups"
        assert ep.frames[expected].shape[0] == ep.length + 1
    if kind == "sampled":
        assert ep.frames["context.latent_noise"].shape == (5, 3, 4)
    if kind in ("graph", "graph_prob", "g2anet"):
        assert ep.transitions["behavior.adjacency"].shape[0] == ep.length


@pytest.mark.parametrize("mode,mean,variance", [
    ("ae", 3.5, 1.), ("sample_mean", 3.5, 13. / 4),
    ("bayesian", 38. / 13, 36. / 13), ("ci", 242. / 97, 468. / 97),
])
def test_gaussian_fusion_matches_hand_calculation(mode, mean, variance):
    module = GaussianConsensus(1, 4, mode)
    values = torch.tensor([[[2.], [5.], [1e6]]])
    logvar = torch.tensor([[[4.], [9.], [1.]]]).log()
    messages = values if mode == "ae" else torch.cat((values, logvar), -1)
    messages.requires_grad_()
    alive = torch.tensor([[True, True, False]])
    groups = torch.tensor([[1, 1, 999]])
    result = module(messages, alive, {"groups": groups})
    assert float(result.diagnostics["group_mu"][0, 1, 0].detach()) == pytest.approx(mean)
    assert result.diagnostics["group_log_var"][0, 1, 0].exp().detach() == pytest.approx(variance)
    assert result.diagnostics["group_valid"].tolist() == [[False, True, False, False]]
    assert not result.features[0, 2].any()
    (result.diagnostics["group_mu"].sum() + result.diagnostics["group_log_var"].sum()).backward()
    assert torch.isfinite(messages.grad).all()
    assert not messages.grad[0, 2].any()
    perm = torch.tensor([2, 0, 1])
    reordered = module(messages[:, perm], alive[:, perm], {"groups": groups[:, perm]})
    torch.testing.assert_close(reordered.features, result.features[:, perm])
    empty = module(messages.detach(), torch.zeros_like(alive), {"groups": groups})
    assert not empty.features.any() and torch.isfinite(empty.diagnostics["group_log_var"]).all()


def test_ci_duplicates_and_singleton_identity():
    for mode, variance in (("ci", 4.), ("bayesian", 2.), ("sample_mean", 2.)):
        module = GaussianConsensus(1, 2, mode)
        raw = torch.tensor([[[3., 1.38629436], [3., 1.38629436]]])
        live = torch.ones(1, 2, dtype=torch.bool)
        merged = module(raw, live, {"groups": torch.zeros_like(live, dtype=torch.long)})
        assert merged.diagnostics["group_log_var"][0, 0, 0].exp() == pytest.approx(variance)
        singleton = module(raw, live, {"groups": torch.tensor([[0, 1]])})
        torch.testing.assert_close(singleton.diagnostics["group_mu"], raw[..., :1])
        torch.testing.assert_close(singleton.diagnostics["group_log_var"], raw[..., 1:])


def test_recorded_groups_change_at_every_time_and_warmup_replays_them():
    class AlternatingGroups(AllTogether):
        def __call__(self, observations, state, alive, timestep):
            groups = torch.tensor([0, timestep % 2], dtype=torch.long)
            return {"groups": groups.masked_fill(~alive, -1)}

    actor, _ = communication_actor("bayesian")
    trainer = make_trainer()
    critic = trainer.learner.critic
    collector = ParallelCollector(TinyParallel, max_steps=4, context_provider=AlternatingGroups())
    ep = collector.collect_episode(actor, critic, seed=1)
    assert ep.frames["context.groups"][:, 1].tolist() == [0, 1, 0, 1, 0]
    learner = MAPPOLearner(actor, critic, torch.optim.Adam(actor.parameters()),
                           torch.optim.Adam(critic.parameters()), burn_in=1)
    _, _, metrics = learner.losses(prepare_ppo_targets({0: ep}), partition_episode(0, ep, length=2))
    assert abs(metrics["approx_kl"]) < 1e-6
    context = frame_context(collate_sequences({0: ep}, partition_episode(0, ep, 2)).frames)
    assert "states" not in context


@pytest.mark.parametrize("source", ["history", "frame", "observation"])
def test_step_sequence_and_message_source(source):
    actor, _ = communication_actor("set", source=source)
    obs = torch.randn(2, 4, 2, 3)
    alive = torch.ones(2, 4, 2, dtype=torch.bool)
    reset = torch.zeros_like(alive)
    state = actor.initial_state(2, 2, "cpu", torch.float32)
    whole = actor.forward_sequence(DecisionInput(obs, alive, reset), state)
    parts = []
    for t in range(4):
        result = actor.step(DecisionInput(obs[:, t], alive[:, t], reset[:, t]), state)
        state = result.state
        parts.append(result.values)
    torch.testing.assert_close(whole.values, torch.stack(parts, 1))
    changed = obs.clone()
    changed[:, 2:] += 100
    future = actor.forward_sequence(DecisionInput(changed, alive, reset),
                                     actor.initial_state(2, 2, "cpu", torch.float32))
    torch.testing.assert_close(future.values[:, :2], whole.values[:, :2])


def test_rl_detach_does_not_detach_auxiliary_representation():
    actor, provider = communication_actor("bayesian", detach_fusion=True)
    obs = torch.randn(1, 3, 2, 3)
    live = torch.ones(1, 3, 2, dtype=torch.bool)
    reset = torch.zeros_like(live)
    context = {"groups": torch.zeros_like(live, dtype=torch.long)}
    inputs = DecisionInput(obs, live, reset, context)
    result = actor(inputs, actor.initial_state(1, 2, "cpu", torch.float32))
    result.values.square().sum().backward()
    assert all(p.grad is None for p in actor.message_encoder.parameters())
    actor.zero_grad(set_to_none=True)
    result = actor(inputs, actor.initial_state(1, 2, "cpu", torch.float32))
    result.diagnostics["group_mu"].square().sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() for p in actor.message_encoder.parameters())


def test_privileged_group_builder_is_explicit_in_episode_metadata():
    class Builder:
        def __call__(self, state):
            return torch.zeros(state.shape[0], 2, dtype=torch.long)

    actor, _ = communication_actor("ae", "q_values")
    provider = StateBuilderContext(Builder())
    ep = ParallelCollector(TinyParallel, max_steps=1, context_provider=provider).collect_episode(actor, seed=1)
    assert ep.metadata["privileged_communication"] is True


def test_dual_path_owns_independent_message_state_and_replays_both_branches():
    from marlite.learning.sequence import initial_states

    actor, provider = communication_actor("bayesian", source="observation")
    actor.message_encoder = nn.Linear(3, 8)
    actor.message_temporal = RecurrentGRU(8, 8)
    actor.message_projection = nn.Linear(8, 8)
    collector = ParallelCollector(lambda: TinyParallel(death=True, mask=True), max_steps=4,
                                   context_provider=provider)
    critic = make_trainer().learner.critic
    episode = collector.collect_episode(actor, critic, seed=19)
    assert "actor_state.message.h" in episode.frames
    learner = MAPPOLearner(actor, critic, torch.optim.Adam(actor.parameters()),
                           torch.optim.Adam(critic.parameters()), burn_in=1)
    episodes = prepare_ppo_targets({0: episode})
    sources = partition_episode(0, episode, length=2)
    _, _, metrics = learner.losses(episodes, sources)
    assert abs(metrics["approx_kl"]) < 1e-6
    source = [sources[1]]
    cached = initial_states(actor, episodes, source, device="cpu", strategy="cached_burnin", burn_in=1)
    exact = initial_states(actor, episodes, source, device="cpu", strategy="prefix_recompute", burn_in=1)
    for key in cached:
        torch.testing.assert_close(cached[key], exact[key])
    assert set(cached) == {"h", "message.h"}
    assert not torch.equal(cached["h"], cached["message.h"])
    learner.update(episodes, sources)
    assert any(p.grad is not None and p.grad.abs().sum() for p in actor.message_temporal.parameters())
