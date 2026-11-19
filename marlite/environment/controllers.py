"""Environment opponents are controllers, not trainable AgentGroup subclasses."""

from copy import deepcopy
import math

import numpy as np
import torch

from marlite.algorithm.sequence_agent_group import DecisionInput
from marlite.util.action_distribution import masked_categorical
from marlite.util.value_learning import masked_action_values
from .heuristic_scores import BattleScores


def legal_actions(availability):
    if hasattr(availability, "n"):
        if getattr(availability, "start", 0) != 0:
            raise ValueError("Opponent actions must be zero-based discrete values")
        return np.arange(availability.n)
    mask = np.asarray(availability)
    if mask.ndim != 1:
        raise ValueError("Opponent availability must be a vector or discrete space")
    indices = np.flatnonzero(mask)
    if not indices.size:
        raise ValueError("A live opponent must have a legal action")
    return indices


class RandomController:
    """Episode-seeded legal action sampling without global NumPy RNG mutation."""

    def __init__(self):
        self.reset()

    def reset(self, seed=None, agents=()):
        self.rng = np.random.default_rng(seed)

    def actions(self, observations, available, alive, state=None):
        return {agent: int(self.rng.choice(legal_actions(available[agent]))) for agent in alive}


class PreyEscapeController:
    """Prefer grid moves with larger summed distance from obstacles/predators."""

    def __init__(self, strategy="greedy", temperature=1.0, top_k=5):
        if strategy not in ("greedy", "probability"):
            raise ValueError("Prey strategy must be greedy or probability")
        if (
            not math.isfinite(temperature)
            or temperature <= 0
            or type(top_k) is not int
            or top_k < 1
        ):
            raise ValueError("Temperature/top_k must be positive")
        self.strategy, self.temperature, self.top_k = strategy, temperature, top_k
        self.reset()

    def reset(self, seed=None, agents=()):
        self.rng = np.random.default_rng(seed)

    def actions(self, observations, available, alive, state=None):
        result = {}
        offsets = np.array([(x, y) for x in (-1, 0, 1) for y in (-1, 0, 1)])
        for agent in alive:
            obs = np.asarray(observations[agent])
            if obs.ndim != 3 or obs.shape[-1] < 4:
                raise ValueError("Prey escape requires H,W,C native observations")
            threats = np.argwhere(obs[..., 0] + obs[..., 3] > 0)
            locations = offsets + np.array(obs.shape[:2]) // 2
            values = np.abs(locations[:, None] - threats[None]).sum((1, 2))
            legal = legal_actions(available[agent])
            if legal.max() >= len(offsets):
                raise ValueError("Prey escape expects the native 9-action schema")
            selected = legal[np.argsort(-values[legal], kind="stable")]
            if self.strategy == "greedy":
                result[agent] = int(selected[0])
            else:
                selected = selected[: self.top_k]
                logits = (values[selected] - values[selected].max()) / self.temperature
                probabilities = np.exp(logits)
                result[agent] = int(
                    self.rng.choice(selected, p=probabilities / probabilities.sum())
                )
        return result


class BattleController:
    """Fixed basic/advanced/probabilistic battle rules with native action indices."""

    def __init__(self, strategy="advanced", temperature=1.0, top_k=8):
        if strategy not in ("basic", "advanced", "probability"):
            raise ValueError("Unknown battle strategy")
        if (
            not math.isfinite(temperature)
            or temperature <= 0
            or type(top_k) is not int
            or top_k < 1
        ):
            raise ValueError("Temperature/top_k must be positive")
        self.strategy = strategy
        self.scorer = BattleScores(strategy=strategy, temperature=temperature, top_k=top_k)
        self.reset()

    def reset(self, seed=None, agents=()):
        self.rng = np.random.default_rng(seed)
        self.scorer.rng = self.rng

    def actions(self, observations, available, alive, state=None):
        if not alive:
            return {}
        history = {agent: np.asarray(observations[agent])[None] for agent in alive}
        if self.strategy == "probability":
            selected, _ = self.scorer.probability_strategy(history, alive)
        else:
            strategy = (
                self.scorer.basic_strategy
                if self.strategy == "basic"
                else self.scorer.advanced_strategy
            )
            selected = strategy(history, alive)
        result = {}
        for agent in alive:
            legal = legal_actions(available[agent])
            if legal.max() >= 21:
                raise ValueError("Battle rules expect the native 21-action schema")
            choice = int(selected[agent])
            result[agent] = choice if choice in legal else int(self.rng.choice(legal))
        return result


class FrozenPolicyController:
    """Run a frozen functional policy with its own per-environment recurrent state."""

    def __init__(self, actor, context_provider=None, device="cpu", stochastic=False, epsilon=0.0):
        if not 0 <= epsilon <= 1:
            raise ValueError("Opponent epsilon must be in [0,1]")
        self.actor = actor.requires_grad_(False).eval().to(device)
        self.device, self.context_provider = torch.device(device), context_provider
        self.stochastic, self.epsilon = stochastic, epsilon

    def reset(self, seed=None, agents=()):
        self.agents, self.state, self.previous_alive, self.timestep = tuple(agents), None, None, 0
        self.rng = torch.Generator()
        self.rng.seed() if seed is None else self.rng.manual_seed(seed)
        self.context = deepcopy(self.context_provider)
        if self.context is not None:
            self.context.reset(seed=seed)

    @torch.no_grad()
    def actions(self, observations, available, alive, state=None):
        if not alive:
            return {}
        zero = np.zeros_like(next(iter(observations.values())))
        obs = torch.as_tensor(
            np.stack([observations.get(a, zero) for a in self.agents]), dtype=torch.float32
        )
        active = torch.tensor([a in alive for a in self.agents], dtype=torch.bool)
        if self.state is None:
            self.state = self.actor.initial_state(1, len(self.agents), self.device, torch.float32)
            self.previous_alive = torch.zeros_like(active)
        reset = active & ~self.previous_alive
        context = (
            {}
            if self.context is None
            else self.context(obs, torch.as_tensor(state), active, self.timestep)
        )
        context.update(self.actor.sample_context(active, self.rng))
        output = self.actor.step(
            DecisionInput(
                obs[None].to(self.device),
                active[None].to(self.device),
                reset[None].to(self.device),
                {key: value[None].to(self.device) for key, value in context.items()},
            ),
            self.state,
        )
        legal = [legal_actions(available[a]) if a in alive else np.array([0]) for a in self.agents]
        width = output.values.shape[-1]
        mask = torch.zeros(len(self.agents), width, dtype=torch.bool, device=self.device)
        for index, choices in enumerate(legal):
            if choices.max() >= width:
                raise ValueError("Opponent policy action head does not fit its environment")
            mask[index, choices] = True
        if self.actor.output_kind == "logits":
            distribution = masked_categorical(
                output.values[0], mask, active_mask=active.to(self.device)
            )
            chosen = (
                torch.multinomial(distribution.probs.cpu(), 1, generator=self.rng).squeeze(-1)
                if self.stochastic
                else distribution.probs.argmax(-1).cpu()
            )
        else:
            values = masked_action_values(
                output.values[0], mask, active_mask=active.to(self.device)
            )
            chosen = values.argmax(-1).cpu()
            if self.stochastic and self.epsilon:
                random = torch.multinomial(mask.cpu().float(), 1, generator=self.rng).squeeze(-1)
                chosen = torch.where(
                    torch.rand(len(self.agents), generator=self.rng) < self.epsilon, random, chosen
                )
        self.state, self.previous_alive = output.state, active
        self.timestep += 1
        return {
            agent: int(chosen[index]) for index, agent in enumerate(self.agents) if agent in alive
        }
