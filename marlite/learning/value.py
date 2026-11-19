"""Sequence value learning: shared execution, distinct algorithm objectives."""

from copy import deepcopy
from dataclasses import dataclass
import math

import torch
from torch import nn

from marlite.algorithm.sequence_agent_group import DecisionInput
from marlite.data import SequenceSlice, collate_sequences
from marlite.util.value_learning import masked_action_values, qtran_constraint_terms
from .backend import ExecutionBackend
from .sequence import initial_states, validate_optimizer_ownership


@dataclass
class LossTerm:
    total: torch.Tensor
    count: torch.Tensor
    weight: float = 1.


def nstep_targets(batch, next_joint, gamma, n_steps, reward_aggregation="sum"):
    """Evaluate each learning transition against its real n-step endpoint.

    The sequence must include a future tail of n_steps-1 transitions. Returns
    are shortened at the real episode boundary, not at the learning boundary.
    """
    if not 0 <= gamma <= 1 or type(n_steps) is not int or n_steps < 1:
        raise ValueError("Invalid n-step discount/horizon")
    if reward_aggregation not in ("sum", "mean"):
        raise ValueError("Reward aggregation must be sum or mean")
    rewards = batch.transitions["rewards"]
    rewards = rewards.sum(-1) if reward_aggregation == "sum" else rewards.mean(-1)
    b, t = rewards.shape
    if next_joint.shape != (b, t + 1):
        raise ValueError("Bootstrap values must cover all T+1 frames")
    positions = torch.arange(t, device=rewards.device)[None].expand(b, -1)
    endpoint = positions.clone()
    accumulated = torch.zeros_like(rewards)
    continuing = batch.filled.clone()
    steps = torch.zeros_like(positions)
    terminal = batch.transitions["terminated"]
    truncated = batch.transitions["truncated"]
    for offset in range(n_steps):
        index = positions + offset
        safe = index.clamp_max(t - 1)
        eligible = (index < t) & batch.filled.gather(1, safe)
        if (batch.loss_mask & continuing & ~eligible).any():
            raise ValueError("Insufficient n-step future tail")
        active = continuing & eligible
        accumulated += gamma ** offset * rewards.gather(1, safe) * active
        steps += active
        endpoint = torch.where(active, safe + 1, endpoint)
        continuing = active & ~terminal.gather(1, safe) & ~truncated.gather(1, safe)
    last = (endpoint - 1).clamp_min(0)
    survivors = batch.frames["alive"].any(-1).gather(1, endpoint)
    bootstrap = (steps > 0) & ~terminal.gather(1, last) & survivors
    discount = torch.pow(torch.full_like(rewards, gamma), steps) * bootstrap
    selected = next_joint.gather(1, endpoint)
    target = accumulated + discount * torch.where(discount != 0, selected, 0)
    return target, endpoint, discount


class QMIXObjective(nn.Module):
    """Double-Q TD objective; local utilities enter the monotonic mixer."""

    def __init__(self):
        super().__init__()
        self.value_network = None

    def terms(self, policy, joint, critic, batch, targets, values=None):
        valid = batch.loss_mask & batch.frames["alive"][:, :-1].any(-1)
        error = joint.values[:, :-1] - targets
        return {"td_loss": LossTerm(error.square()[valid].sum(), valid.sum())}


class QPLEXObjective(QMIXObjective):
    """Duplex-dueling TD plus the transformation's per-frame regularizer."""

    def terms(self, policy, joint, critic, batch, targets, values=None):
        terms = super().terms(policy, joint, critic, batch, targets)
        valid = batch.loss_mask & batch.frames["alive"][:, :-1].any(-1)
        penalty = joint.diagnostics["attention_penalty"][:, :-1]
        terms["attention_regularization"] = LossTerm(penalty[valid].sum(), valid.sum())
        return terms


class QTRANObjective(QMIXObjective):
    """QTRAN-alt TD, optimality and counterfactual non-optimality constraints."""

    def __init__(self, value_network, lambda_opt=1., lambda_nopt=1., mask_optimal_samples=True):
        super().__init__()
        if any(not math.isfinite(w) or w < 0 for w in (lambda_opt, lambda_nopt)):
            raise ValueError("QTRAN constraint weights must be finite and nonnegative")
        self.value_network = value_network
        self.lambda_opt, self.lambda_nopt = lambda_opt, lambda_nopt
        self.mask_optimal_samples = mask_optimal_samples

    def terms(self, policy, joint, critic, batch, targets, values=None):
        terms = super().terms(policy, joint, critic, batch, targets)
        alive = batch.frames["alive"][:, :-1] & batch.loss_mask[..., None]
        q = policy.values[:, :-1].flatten(0, 1)
        features = policy.features[:, :-1].flatten(0, 1)
        actions = batch.transitions["actions"].flatten(0, 1)
        counterfactual = joint.diagnostics["counterfactual"][:, :-1].flatten(0, 1)
        available = batch.frames["available_actions"][:, :-1].flatten(0, 1)
        opt, opt_count, nopt, nopt_count = qtran_constraint_terms(
            q, features, actions, counterfactual, values.flatten(), critic.core,
            alive.flatten(0, 1), available, self.mask_optimal_samples)
        terms["optimality_loss"] = LossTerm(opt, opt_count, self.lambda_opt)
        terms["nonoptimality_loss"] = LossTerm(nopt, nopt_count, self.lambda_nopt)
        return terms


class ValueLearner:
    """Shared update executor with explicit objective and mixer composition.

    Target actor/mixer own independent states and never consume behavior caches.
    Each objective declares its own loss populations. QTRAN's required V network
    belongs to the objective, not to the target network or optional SSL heads.
    """

    def __init__(self, actor, critic, objective, actor_optimizer, critic_optimizer, *,
                 objective_optimizer=None, backend=None, gamma=.99, n_steps=1,
                 reward_aggregation="sum", burn_in=32, state_strategy="zero_burnin",
                 target_mode="hard", target_interval=200, target_tau=.005, max_grad_norm=5.):
        if actor.output_kind != "q_values":
            raise ValueError("Value learners require individual Q values")
        if not 0 <= gamma <= 1 or type(n_steps) is not int or n_steps < 1:
            raise ValueError("Invalid TD discount/horizon")
        if reward_aggregation not in ("sum", "mean"):
            raise ValueError("Invalid reward aggregation")
        if state_strategy not in ("zero_burnin", "prefix_recompute"):
            raise ValueError("Off-policy online/target states must be independently reconstructed")
        if type(burn_in) is not int or burn_in < 0:
            raise ValueError("Invalid burn-in")
        if target_mode not in ("hard", "ema") or type(target_interval) is not int or target_interval < 1:
            raise ValueError("Invalid target update policy")
        if not math.isfinite(target_tau) or not 0 < target_tau <= 1:
            raise ValueError("target_tau must lie in (0,1]")
        if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
            raise ValueError("Invalid gradient clipping")
        if (objective.value_network is None) != (objective_optimizer is None):
            raise ValueError("QTRAN requires a V optimizer; other objectives do not")
        validate_optimizer_ownership(((actor, actor_optimizer), (critic, critic_optimizer),
                                      (objective, objective_optimizer)))
        self.actor, self.critic, self.objective = actor, critic, objective
        self.actor_optimizer, self.critic_optimizer = actor_optimizer, critic_optimizer
        self.objective_optimizer = objective_optimizer
        self.backend = backend or ExecutionBackend(next(actor.parameters()).device)
        self.gamma, self.n_steps, self.reward_aggregation = gamma, n_steps, reward_aggregation
        self.burn_in, self.state_strategy = burn_in, state_strategy
        self.target_mode, self.target_interval, self.target_tau = target_mode, target_interval, target_tau
        self.max_grad_norm = max_grad_norm
        modules = (self.actor, self.critic, self.objective)
        identities = [set(map(id, module.parameters())) for module in modules]
        if any(identities[i] & identities[j] for i in range(3) for j in range(i)):
            raise ValueError("Actor, mixer and objective must own disjoint parameters")
        for module in modules:
            module.to(self.backend.device)
            self.backend.synchronize(module)
        self.target_actor = deepcopy(actor).requires_grad_(False).eval()
        self.target_critic = deepcopy(critic).requires_grad_(False).eval()
        self.updates = self.actor_updates = self.critic_updates = self.objective_updates = 0

    @property
    def collection_critic(self):
        return None

    def _initial(self, model, episodes, sources, *, team=False):
        return initial_states(model, episodes, sources, device=self.backend.device,
                              strategy=self.state_strategy, burn_in=self.burn_in, team=team)

    def losses(self, episodes, sources):
        empty = not sources
        if empty:
            if not self.backend.distributed:
                raise ValueError("An update needs at least one sequence")
            sources = [SequenceSlice(next(iter(episodes)), 0, 1)]
        # Keep future transitions for n-step endpoints, but do not train on them.
        batch = collate_sequences(episodes, [SequenceSlice(s.episode_id, s.start, s.length,
                                                           future=self.n_steps - 1) for s in sources])
        batch = batch.to(self.backend.device)
        if empty:
            batch.loss_mask.zero_()
        valid_frames = torch.cat((batch.filled[:, :1], batch.filled), 1)
        alive = batch.frames["alive"] & valid_frames[..., None]
        inputs = DecisionInput(batch.frames["observations"], alive, batch.frames["agent_reset"])
        available, states = batch.frames["available_actions"], batch.frames["states"]
        team_valid = valid_frames & alive.any(-1)
        reset = torch.zeros_like(valid_frames)
        reset[:, 0] = torch.tensor([s.start == 0 for s in sources], device=self.backend.device)
        online = self.actor.forward_sequence(inputs, self._initial(self.actor, episodes, sources))
        actions = torch.cat((batch.transitions["actions"],
                             torch.zeros_like(batch.transitions["actions"][:, :1])), 1)
        current = self.critic.forward_sequence(online, states, actions, alive, available,
                                               self._initial(self.critic, episodes, sources, team=True),
                                               team_valid, reset)
        with torch.no_grad():
            greedy = masked_action_values(online.values.detach(), available, alive).argmax(-1)
            target = self.target_actor.forward_sequence(inputs, self._initial(self.target_actor, episodes, sources))
            target_joint = self.target_critic.forward_sequence(
                target, states, greedy, alive, available,
                self._initial(self.target_critic, episodes, sources, team=True), team_valid, reset)
            targets, _, _ = nstep_targets(batch, target_joint.values, self.gamma, self.n_steps,
                                          self.reward_aggregation)
        values = None
        if self.objective.value_network is not None:
            v_net = self.objective.value_network
            values = v_net.forward_sequence(states[:, :-1], self._initial(v_net, episodes, sources, team=True),
                                            team_valid[:, :-1], reset[:, :-1]).values
        terms = self.objective.terms(online, current, self.critic, batch, targets, values)
        total = online.values.sum() * 0  # Empty ranks still own the actor graph.
        metrics = {}
        for name, term in terms.items():
            count = self.backend.sum(term.count.to(device=self.backend.device, dtype=torch.float32))
            if term.weight:
                total = total + term.weight * term.total / count.clamp_min(1)
            metrics[name] = float(self.backend.sum(term.total) / count.clamp_min(1))
            metrics[name + "_count"] = float(count)
        metrics["team_steps"] = metrics["td_loss_count"]
        return total, metrics

    def update(self, episodes, sources):
        modules = [self.actor, self.critic, self.objective]
        optimizers = [self.actor_optimizer, self.critic_optimizer, self.objective_optimizer]
        for module, optimizer in zip(modules, optimizers):
            module.train()
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
        loss, metrics = self.losses(episodes, sources)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite value-learning objective")
        loss.backward()
        for module in modules:
            self.backend.reduce_gradients(module.parameters())
        counters = ("actor_updates", "critic_updates", "objective_updates")
        for module, optimizer, counter in zip(modules, optimizers, counters):
            if optimizer is not None and metrics["team_steps"] and any(p.grad is not None for p in module.parameters()):
                torch.nn.utils.clip_grad_norm_(module.parameters(), self.max_grad_norm)
                optimizer.step()
                setattr(self, counter, getattr(self, counter) + 1)
        if metrics["team_steps"]:
            self.updates += 1
            if self.updates % self.target_interval == 0:
                self.refresh_targets()
        return metrics

    @torch.no_grad()
    def refresh_targets(self, *, hard=False):
        for target, online in ((self.target_actor, self.actor), (self.target_critic, self.critic)):
            for name, value in target.state_dict().items():
                source = online.state_dict()[name]
                if hard or self.target_mode == "hard" or not value.is_floating_point():
                    value.copy_(source)
                else:
                    value.lerp_(source, self.target_tau)

    def state_dict(self):
        result = {name: getattr(self, name).state_dict() for name in
                  ("actor", "critic", "objective", "target_actor", "target_critic",
                   "actor_optimizer", "critic_optimizer")}
        result["objective_optimizer"] = (None if self.objective_optimizer is None else
                                          self.objective_optimizer.state_dict())
        result["counters"] = {name: getattr(self, name) for name in
                              ("updates", "actor_updates", "critic_updates", "objective_updates")}
        return result

    def load_state_dict(self, state):
        for name in ("actor", "critic", "objective", "target_actor", "target_critic",
                     "actor_optimizer", "critic_optimizer", "objective_optimizer"):
            obj = getattr(self, name)
            if obj is not None:
                obj.load_state_dict(state[name])
        for name, value in state["counters"].items():
            setattr(self, name, value)
