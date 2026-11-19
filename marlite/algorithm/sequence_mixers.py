"""Sequence joint-value networks composed from configurable PyTorch modules."""

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from marlite.util.value_learning import masked_action_values, joint_q_from_counterfactual


@dataclass
class JointOutput:
    values: Tensor
    state: dict[str, Tensor]
    diagnostics: dict[str, Tensor] = field(default_factory=dict)


class MonotonicMixing(nn.Module):
    """QMIX's positive hypernetwork mixing core, independent of temporal layout."""

    def __init__(self, n_agents, hidden_dim, w1, b1, w2, b2):
        super().__init__()
        self.n_agents, self.hidden_dim = n_agents, hidden_dim
        self.w1, self.b1, self.w2, self.b2 = w1, b1, w2, b2

    def forward(self, selected_q, state):
        leading = selected_q.shape[:-1]
        q = selected_q.reshape(-1, 1, self.n_agents)
        s = state.reshape(q.shape[0], -1)
        w1 = self.w1(s).abs().reshape(-1, self.n_agents, self.hidden_dim)
        b1 = self.b1(s).reshape(-1, 1, self.hidden_dim)
        hidden = F.elu(torch.bmm(q, w1) + b1)
        w2 = self.w2(s).abs().reshape(-1, self.hidden_dim, 1)
        b2 = self.b2(s).reshape(-1, 1, 1)
        return (torch.bmm(hidden, w2) + b2).reshape(leading)


class DuplexTransformation(nn.Module):
    """QPLEX transformation; return per-sample regularization before reduction.

    Selectors/keys are per-head configurable modules. Optional nonlinear keys
    consume [state, detached feasible maximum Q_i], never the sampled action Q.
    """

    def __init__(self, n_agents, embed_dim, selectors, keys, bias, *,
                 head_weight=None, nonlinear=False, regularization=0.):
        super().__init__()
        if not selectors or len(selectors) != len(keys):
            raise ValueError("Transformation needs equal nonempty selector/key lists")
        self.n_agents, self.embed_dim = n_agents, embed_dim
        self.selectors, self.keys = nn.ModuleList(selectors), nn.ModuleList(keys)
        self.bias, self.head_weight = bias, head_weight
        self.nonlinear, self.regularization = nonlinear, regularization
        if regularization < 0:
            raise ValueError("Attention regularization must be nonnegative")

    def forward(self, maximum_q, state):
        n = state.shape[0]
        key_input = (torch.cat((state[:, None].expand(-1, self.n_agents, -1),
                                maximum_q.detach()[..., None]), -1) if self.nonlinear else state)
        weights, penalties = [], []
        for selector, key in zip(self.selectors, self.keys):
            query = selector(state)
            encoded = key(key_input).reshape(n, self.n_agents, self.embed_dim)
            logits = torch.bmm(query[:, None], encoded.transpose(1, 2)).squeeze(1)
            weights.append((logits / self.embed_dim ** .5).softmax(-1))
            penalties.append(logits.square().mean(-1))
        attention = torch.stack(weights, 1)
        if self.head_weight is not None:
            attention = attention * (self.head_weight(state).abs() + 1e-10)[..., None]
        return (attention.sum(1) + 1e-10,
                self.bias(state).reshape(n, 1).expand(-1, self.n_agents),
                self.regularization * torch.stack(penalties, 1).sum(1))


class JointAdvantageWeights(nn.Module):
    """Positive QPLEX joint-action-dependent weights; sum across attention heads."""

    def __init__(self, keys, agent_gates, action_gates):
        super().__init__()
        if not keys or not len(keys) == len(agent_gates) == len(action_gates):
            raise ValueError("Joint attention needs matching nonempty head lists")
        self.keys = nn.ModuleList(keys)
        self.agent_gates, self.action_gates = nn.ModuleList(agent_gates), nn.ModuleList(action_gates)

    def forward(self, state, joint_action):
        data = torch.cat((state, joint_action), -1)
        return torch.stack([(key(state).abs() + 1e-10) * gate(state).sigmoid()
                            * action(data).sigmoid() for key, gate, action in
                            zip(self.keys, self.agent_gates, self.action_gates)], 0).sum(0)


class SequenceQMixer(nn.Module):
    """Team temporal state encoder followed by monotonic scalar mixing."""

    def __init__(self, state_encoder, mixing):
        super().__init__()
        self.state_encoder, self.mixing = state_encoder, mixing

    def initial_state(self, batch_size, device, dtype):
        return self.state_encoder.initial_state(batch_size, device, dtype)

    def encode_sequence(self, states, initial_state, valid, reset):
        return self.state_encoder.encode_sequence(states, initial_state, valid, reset)

    def forward_sequence(self, policy, states, actions, alive, available, initial_state, valid, reset):
        encoded = self.encode_sequence(states, initial_state, valid, reset)
        selected = policy.values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        chosen = selected.masked_fill(~alive, 0)
        return JointOutput(self.mixing(chosen, encoded.features).masked_fill(~valid, 0), encoded.state)


class SequenceDuplexMixer(nn.Module):
    """QPLEX joint value from full per-agent Q tables and actual joint actions."""

    def __init__(self, state_encoder, transformation, joint_attention, value_weight, value_bias,
                 *, action_dim, n_agents, weighted_head=True, minus_one=True):
        super().__init__()
        self.state_encoder = state_encoder
        self.transformation, self.joint_attention = transformation, joint_attention
        self.value_weight, self.value_bias = value_weight, value_bias
        self.action_dim, self.n_agents = action_dim, n_agents
        self.weighted_head, self.minus_one = weighted_head, minus_one

    def initial_state(self, batch_size, device, dtype):
        return self.state_encoder.initial_state(batch_size, device, dtype)

    def encode_sequence(self, states, initial_state, valid, reset):
        return self.state_encoder.encode_sequence(states, initial_state, valid, reset)

    def forward_sequence(self, policy, states, actions, alive, available, initial_state, valid, reset):
        encoded = self.encode_sequence(states, initial_state, valid, reset)
        leading = alive.shape[:-1]
        flat_state = encoded.features.flatten(0, 1)
        q = policy.values.flatten(0, 1)
        live = alive.flatten(0, 1)
        feasible = masked_action_values(q, available.flatten(0, 1), live)
        maximum = feasible.max(-1).values
        effective = actions.flatten(0, 1).masked_fill(~live, 0)
        chosen = q.gather(-1, effective[..., None]).squeeze(-1).masked_fill(~live, 0)
        weights, bias, regularizer = self.transformation(maximum.detach(), flat_state)
        transformed = weights * chosen + bias
        transformed_max = weights * maximum + bias
        if self.weighted_head:
            head_w, head_b = self.value_weight(flat_state).abs() + 1e-10, self.value_bias(flat_state)
            transformed = head_w * transformed + head_b
            transformed_max = head_w * transformed_max + head_b
        transformed = transformed.masked_fill(~live, 0)
        transformed_max = transformed_max.masked_fill(~live, 0)
        local_advantage = (transformed - transformed_max).detach()
        onehot = F.one_hot(effective, num_classes=self.action_dim).to(flat_state).flatten(1)
        attention = self.joint_attention(flat_state, onehot) - float(self.minus_one)
        value = transformed.sum(-1)
        advantage = (attention * local_advantage).sum(-1)
        return JointOutput((value + advantage).reshape(leading).masked_fill(~valid, 0), encoded.state,
                           {"attention_penalty": regularizer.reshape(leading),
                            "value": value.reshape(leading), "advantage": advantage.reshape(leading)})


class CounterfactualJoint(nn.Module):
    """QTRAN-alt: agent i's counterfactual vector excludes its own action."""

    def __init__(self, phi, psi, head, action_dim):
        super().__init__()
        self.phi, self.psi, self.head = phi, psi, head
        self.action_dim = action_dim

    def forward(self, features, actions, alive_mask=None):
        alive = torch.ones_like(actions, dtype=torch.bool) if alive_mask is None else alive_mask
        actions = actions.masked_fill(~alive, 0)
        features = features.masked_fill(~alive[..., None], 0)
        onehot = F.one_hot(actions.long(), num_classes=self.action_dim).to(features)
        phi = self.phi(torch.cat((features, onehot), -1)).masked_fill(~alive[..., None], 0)
        count = alive.sum(-1, keepdim=True).clamp_min(1)[..., None]
        vector = self.psi(features) + phi.sum(-2, keepdim=True) / count - phi / count
        return {"q_per_action": self.head(vector).masked_fill(~alive[..., None], 0)}


class SequenceCounterfactualMixer(nn.Module):
    """QTRAN joint network reads agent histories, not the global state."""

    def __init__(self, core):
        super().__init__()
        self.core = core

    def initial_state(self, batch_size, device, dtype):
        return {}

    def encode_sequence(self, states, initial_state, valid, reset):
        # No centralized recurrent state: histories already live in policy features.
        return JointOutput(states, {})

    def forward_sequence(self, policy, states, actions, alive, available, initial_state, valid, reset):
        cf = self.core(policy.features.flatten(0, 1), actions.flatten(0, 1),
                       alive_mask=alive.flatten(0, 1))["q_per_action"]
        joint = joint_q_from_counterfactual(cf, actions.flatten(0, 1), alive.flatten(0, 1))
        return JointOutput(joint.reshape(valid.shape).masked_fill(~valid, 0), {},
                           {"counterfactual": cf.reshape(*alive.shape, -1)})
