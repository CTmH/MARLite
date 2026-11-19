"""Small runnable defaults; YAML can replace any composed module explicitly."""

from torch import nn

from marlite.algorithm.sequence_agent_group import SequenceAgentGroup
from marlite.algorithm.sequence_value import SequenceValue, TeamSequenceEncoder
from marlite.algorithm.sequence_mixers import (
    MonotonicMixing,
    DuplexTransformation,
    JointAdvantageWeights,
    SequenceQMixer,
    SequenceDuplexMixer,
    CounterfactualJoint,
    SequenceCounterfactualMixer,
)
from marlite.algorithm.temporal import RecurrentGRU
from marlite.communication.group_set import GroupSetCommunication, QuantizedCommunication
from marlite.algorithm.model.group_communication import GroupSetFusion, ScalarMessageCodec
from marlite.learning.value import QMIXObjective, QPLEXObjective, QTRANObjective


def encoder(size, hidden):
    return nn.Sequential(nn.Flatten(1), nn.Linear(size, hidden), nn.GELU())


def actor(spec, hidden=64, *, output_kind="logits", quantized=False, heads=4):
    communication, readout, action_width = None, None, hidden
    if quantized:
        aggregation = GroupSetCommunication(
            GroupSetFusion(hidden, heads, 8, 16),
            nn.MultiheadAttention(hidden, heads, dropout=0.0, batch_first=True),
            header_bytes=9,
        )
        communication = QuantizedCommunication(
            ScalarMessageCodec(hidden, 32, 4), aggregation, header_bytes=9
        )
        readout = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU())
    return SequenceAgentGroup(
        encoder(spec.dimensions["observation_size"], hidden),
        RecurrentGRU(hidden, hidden),
        nn.Linear(action_width, spec.action_dim),
        output_kind,
        communication=communication,
        readout=readout,
    )


def value(spec, hidden=64):
    return SequenceValue(
        encoder(spec.dimensions["state_size"], hidden),
        RecurrentGRU(hidden, hidden),
        nn.Linear(hidden, 1),
    )


def mixer(recipe, spec, actor, *, hidden=64, mixing=32, heads=4):
    n, a = len(spec.agent_ids), spec.action_dim
    state = None
    if recipe != "qtran":
        state = TeamSequenceEncoder(
            encoder(spec.dimensions["state_size"], hidden), RecurrentGRU(hidden, hidden)
        )
    if recipe == "qmix":
        critic = SequenceQMixer(
            state,
            MonotonicMixing(
                n,
                mixing,
                nn.Linear(hidden, n * mixing),
                nn.Linear(hidden, mixing),
                nn.Linear(hidden, mixing),
                nn.Linear(hidden, 1),
            ),
        )
    elif recipe == "qplex":
        transform = DuplexTransformation(
            n,
            mixing,
            [nn.Linear(hidden, mixing) for _ in range(heads)],
            [nn.Linear(hidden, n * mixing) for _ in range(heads)],
            nn.Linear(hidden, 1),
            head_weight=nn.Linear(hidden, heads),
        )
        attention = JointAdvantageWeights(
            [nn.Linear(hidden, 1) for _ in range(heads)],
            [nn.Linear(hidden, n) for _ in range(heads)],
            [nn.Linear(hidden + n * a, n) for _ in range(heads)],
        )
        critic = SequenceDuplexMixer(
            state,
            transform,
            attention,
            nn.Linear(hidden, n),
            nn.Linear(hidden, n),
            action_dim=a,
            n_agents=n,
        )
    else:
        feature_width = getattr(actor.action_head, "in_features", hidden)
        core = CounterfactualJoint(
            nn.Linear(feature_width + a, hidden),
            nn.Linear(feature_width, hidden),
            nn.Linear(hidden, a),
            a,
        )
        critic = SequenceCounterfactualMixer(core)
    return critic


def objective(recipe, spec, hidden=64):
    if recipe == "qmix":
        return QMIXObjective()
    if recipe == "qplex":
        return QPLEXObjective()
    return QTRANObjective(value(spec, hidden))
