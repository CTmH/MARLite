"""Private encoders, temporal state and heads compose with all sequence learners."""

from copy import deepcopy

import pytest
import torch
from torch import nn

from marlite.algorithm.sequence_agent_group import SequenceAgentGroup, DecisionInput
from marlite.algorithm.sequence_value import SequenceValue
from marlite.algorithm.temporal import RecurrentGRU, CausalAttention, WindowEncoder
from marlite.algorithm.routing import RoutedFrames, RoutedTemporal, RoutedHead
from marlite.algorithm.components import Permute
from marlite.algorithm.adapters import GaussianParameters, MaskedPool
from marlite.collection.parallel import ParallelCollector
from marlite.data import partition_episode
from marlite.learning.mappo import MAPPOLearner, prepare_ppo_targets
from marlite.learning.value import ValueLearner
from marlite.environment.padding import PadAgentView
from test.test_environment_adapters import HeterogeneousEnv
from test.test_sequence_mappo import assert_tree_equal
from test.test_sequence_value import make_value_learner


ROLES = ["small", "large", "small"]


def routed_learner(algorithm="mappo"):
    torch.manual_seed(72)
    frames = RoutedFrames(
        ROLES, {"small": nn.Linear(2, 6), "large": nn.Linear(4, 6)}, 6, {"small": 2, "large": 4}
    )
    temporal = RoutedTemporal(
        ROLES,
        {"small": RecurrentGRU(6, 5), "large": CausalAttention(6, 6, heads=2, context_length=3)},
        6,
    )
    head = RoutedHead(
        ROLES, {"small": nn.Linear(5, 2), "large": nn.Linear(6, 3)}, 6, 3, {"small": 5, "large": 6}
    )
    actor = SequenceAgentGroup(
        frames, temporal, head, "logits" if algorithm == "mappo" else "q_values"
    )
    if algorithm == "mappo":
        critic = SequenceValue(nn.Linear(6, 8), RecurrentGRU(8, 8), nn.Linear(8, 1))
        return MAPPOLearner(
            actor,
            critic,
            torch.optim.AdamW(actor.parameters(), lr=0.001, weight_decay=0.1),
            torch.optim.Adam(critic.parameters()),
            burn_in=1,
        )
    base = make_value_learner(algorithm, state_dim=6, action_dim=3, n_agents=3)
    if algorithm == "qtran":
        base.critic.core.phi = nn.Linear(9, 8)
        base.critic.core.psi = nn.Linear(6, 8)
    return ValueLearner(
        actor,
        base.critic,
        base.objective,
        torch.optim.AdamW(actor.parameters(), lr=0.001, weight_decay=0.1),
        torch.optim.Adam(base.critic.parameters()),
        objective_optimizer=base.objective_optimizer,
        n_steps=2,
        burn_in=1,
    )


def routed_data(learner):
    episodes = {}
    for index, selection in enumerate(("small", "large")):
        collector = ParallelCollector(
            lambda: PadAgentView(HeterogeneousEnv(selection)), max_steps=3
        )
        episodes[index] = collector.collect_episode(
            learner.actor,
            learner.critic if learner.actor.output_kind == "logits" else None,
            seed=20 + index,
        )
    if learner.actor.output_kind == "logits":
        episodes = prepare_ppo_targets(episodes)
    sources = [
        source
        for key, episode in episodes.items()
        for source in partition_episode(key, episode, length=2)
    ]
    return episodes, sources


@pytest.mark.parametrize("algorithm", ["mappo", "qmix", "qplex", "qtran"])
def test_routed_models_replay_train_and_restore_with_independent_state(algorithm):
    learner = routed_learner(algorithm)
    episodes, sources = routed_data(learner)
    if algorithm == "mappo":
        assert "actor_state.small/h" in episodes[0].state_snapshots
        assert "actor_state.large/cache" in episodes[0].state_snapshots
        _, _, metrics = learner.losses(episodes, sources)
        assert abs(metrics["approx_kl"]) < 1e-6
    assert all(torch.isfinite(torch.tensor(v)) for v in learner.update(episodes, sources).values())
    checkpoint = deepcopy(learner.state_dict())
    reference = learner.update(episodes, sources)
    restored = routed_learner(algorithm)
    restored.load_state_dict(checkpoint)
    assert reference == restored.update(episodes, sources)
    assert_tree_equal(learner.state_dict(), restored.state_dict())
    # No large-role sample: its existing momentum and weight decay must not drift.
    before = {
        name: value.clone()
        for name, value in learner.actor.state_dict().items()
        if ".large." in name
    }
    learner.update(episodes, [source for source in sources if source.episode_id == 0])
    for name, value in before.items():
        torch.testing.assert_close(value, learner.actor.state_dict()[name], rtol=0, atol=0)
    assert all(p.grad is None for p in learner.actor.action_head.models["large"].parameters())


def test_routed_sequence_step_equivalence_and_life_resets():
    actor = routed_learner().actor
    obs = torch.randn(2, 4, 3, 8)
    valid = torch.ones(2, 4, 3, dtype=torch.bool)
    valid[0, 1:3, 1] = False
    reset = torch.zeros_like(valid)
    reset[:, 0] = True
    reset[0, 3, 1] = True
    state = actor.initial_state(2, 3, "cpu", torch.float32)
    whole = actor(DecisionInput(obs, valid, reset), state)
    rows = []
    for time in range(4):
        result = actor.step(DecisionInput(obs[:, time], valid[:, time], reset[:, time]), state)
        rows.append(result.values)
        state = result.state
    torch.testing.assert_close(whole.values, torch.stack(rows, 1))
    for key in state:
        torch.testing.assert_close(whole.state[key], state[key])


def test_convolutional_window_model_uses_every_causal_timestep():
    model = nn.Sequential(
        Permute([0, 2, 1]), nn.Conv1d(2, 3, 1), nn.AdaptiveAvgPool1d(1), nn.Flatten(1)
    )
    encoder = WindowEncoder(model, 2, 3, 4, use_padding_mask=False)
    inputs = torch.randn(1, 4, 2, 2)
    state = encoder.initial_state(1, 2, "cpu", torch.float32)
    output = encoder.forward_sequence(inputs, state)
    changed = inputs.clone()
    changed[:, 2:] += 100
    other = encoder.forward_sequence(changed, state)
    torch.testing.assert_close(output.features[:, :2], other.features[:, :2])
    output.features.sum().backward()
    assert model[1].weight.grad.abs().sum() > 0


def test_mask_and_gaussian_output_adapters_have_explicit_conventions():
    class Pool(nn.Module):
        def forward(self, inputs, padding_mask=None):
            live = (~padding_mask)[..., None]
            mean = inputs.masked_fill(~live, 0).sum(1) / live.sum(1).clamp_min(1)
            return mean, torch.zeros_like(mean)

    inputs = torch.tensor([[[2.0, 4.0], [999.0, 999.0]], [[1.0, 2.0], [3.0, 4.0]]])
    alive = torch.tensor([[True, False], [False, False]])
    result = MaskedPool(Pool(), gaussian_output=True)(inputs, alive)
    torch.testing.assert_close(result, torch.tensor([[2.0, 4.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]))

    class Moments(nn.Module):
        def forward(self, x):
            return x, x * 0

    assert GaussianParameters(Moments())(inputs).shape == (2, 2, 4)


def test_routed_private_models_match_global_update_with_empty_rank(tmp_path):
    from test.sequence_test_tools import run_torchrun

    learner = routed_learner()
    episodes, sources = routed_data(learner)
    selected = [s for s in sources if s.episode_id == 0][:1]
    learner.update(episodes, selected)
    code = """
import os
from pathlib import Path
import torch
import torch.distributed as dist
from marlite.training.distributed import initialize_backend
from test.test_model_routing import routed_learner, routed_data
backend = initialize_backend("cpu", timeout_seconds=45)
try:
    learner = routed_learner()
    episodes, sources = routed_data(learner)
    selected = [s for s in sources if s.episode_id == 0][:1]
    learner.update(episodes, selected if backend.rank == 0 else [])
    if backend.rank == 0:
        torch.save(learner.state_dict(), Path(os.environ["TEST_RUN_DIR"]) / "result.pt")
finally:
    dist.destroy_process_group()
"""
    run_torchrun(tmp_path, code)
    actual = torch.load(tmp_path / "result.pt", weights_only=True)
    for component in ("actor", "critic"):
        for name, expected in learner.state_dict()[component].items():
            torch.testing.assert_close(actual[component][name], expected, atol=2e-6, rtol=2e-5)


def test_existing_masked_probability_encoder_uses_live_mask_and_tensor_output():
    from marlite.algorithm.model.resnet import ResAttMaskedProbEnc

    model = MaskedPool(
        ResAttMaskedProbEnc(3, 8, 8, 2, 4, dropout=0.0),
        mask_argument="alive_mask",
        true_is_valid=True,
    )
    inputs = torch.randn(2, 3, 3)
    alive = torch.tensor([[True, False, True], [False, False, False]])
    result = model(inputs, alive)
    assert result.shape == (2, 8) and result[1].eq(0).all()
    changed = inputs.clone()
    changed[~alive] = 999
    torch.testing.assert_close(result, model(changed, alive))
