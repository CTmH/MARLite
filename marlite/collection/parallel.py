"""Collect complete ParallelEnv episodes under one immutable policy version."""

from copy import deepcopy

import numpy as np
import torch

from marlite.algorithm.sequence_agent_group import DecisionInput
from marlite.data import Episode
from marlite.util.action_distribution import masked_categorical
from marlite.util.value_learning import masked_action_values


def _snapshot(state):
    # Clone: some recurrent implementations can return views of earlier state.
    return {key: value[0].detach().cpu().clone() for key, value in state.items()}


class ParallelCollector:
    """Synchronous collection reference used by both training and evaluation.

    The factory owns simulator construction/wrapping. Tensor conversion happens
    once at this boundary; observation dictionaries may contain action_mask.
    """

    def __init__(self, env_factory, max_steps=200, device="cpu", context_provider=None):
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.env_factory, self.max_steps, self.device = env_factory, max_steps, torch.device(device)
        self.context_provider = context_provider

    @torch.no_grad()
    def collect_episode(self, actor, critic=None, *, seed, policy_version=0, stochastic=True, epsilon=0., record_values=True):
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0,1]")
        if actor.output_kind == "logits" and critic is None and record_values:
            raise ValueError("PPO collection requires a value model")
        if not record_values and critic is not None:
            raise ValueError("Value-free inference must not supply a critic")
        actor_device = next(actor.parameters()).device
        critic_device = next(critic.parameters()).device if critic is not None else None
        actor_mode, critic_mode = actor.training, critic.training if critic is not None else None
        env = self.env_factory()
        try:
            actor.eval().to(self.device)
            if critic is not None:
                critic.eval().to(self.device)
            return self._collect(env, actor, critic, seed, policy_version, stochastic, epsilon)
        finally:
            env.close()
            actor.to(actor_device).train(actor_mode)
            if critic is not None:
                critic.to(critic_device).train(critic_mode)

    def _collect(self, env, actor, critic, seed, policy_version, stochastic, epsilon):
        observations, infos = env.reset(seed=seed)
        provider = deepcopy(self.context_provider)
        if provider is not None and hasattr(provider, "reset"):
            provider.reset(seed=seed)
        agents = tuple(env.possible_agents)
        if not agents or not observations:
            raise ValueError("Reset must expose at least one observation")
        sizes = [int(env.action_space(a).n) for a in agents]
        if len(set(sizes)) != 1:
            raise ValueError("Current collector requires one shared discrete action dimension")
        action_dim = sizes[0]
        sample = next(iter(observations.values()))
        sample = sample["observation"] if isinstance(sample, dict) else sample
        zero = np.zeros_like(sample, dtype=np.float32)
        rng = torch.Generator(device="cpu")
        if seed is None:
            rng.seed()
        else:
            rng.manual_seed(seed)
        frames, transitions = {}, {}
        actor_state = actor.initial_state(1, len(agents), self.device, torch.float32)
        critic_state = critic.initial_state(1, self.device, torch.float32) if critic is not None else {}
        alive = {a: a in env.agents for a in agents}
        previous_alive = {a: False for a in agents}
        dead = {a: False for a in agents}
        last_state = None
        terminal = False
        episode_end = False

        def append(target, name, value):
            target.setdefault(name, []).append(value.detach().cpu().clone())

        for timestep in range(self.max_steps + 1):
            obs_values, masks = [], []
            for agent in agents:
                raw = observations.get(agent)
                if alive[agent] and raw is None:
                    raise ValueError("Missing real observation for a surviving agent")
                if isinstance(raw, dict):
                    obs_values.append(raw["observation"])
                    masks.append(raw.get("action_mask", np.ones(action_dim, dtype=bool)))
                else:
                    obs_values.append(zero if raw is None else raw)
                    masks.append(np.ones(action_dim, dtype=bool))
            obs = torch.as_tensor(np.stack(obs_values), dtype=torch.float32)
            active = torch.tensor([alive[a] for a in agents], dtype=torch.bool)
            reset = torch.tensor([alive[a] and not previous_alive[a] for a in agents], dtype=torch.bool)
            available = torch.as_tensor(np.stack(masks), dtype=torch.bool)
            try:
                state = torch.as_tensor(np.asarray(env.state()), dtype=torch.float32)
            except Exception:
                if not terminal or last_state is None:
                    raise
                state = last_state
            context = {} if provider is None else provider(obs, state, active, timestep)
            context = {key: torch.as_tensor(value).detach().cpu() for key, value in context.items()}
            sampled = actor.sample_context(active, rng)
            if context.keys() & sampled.keys():
                raise ValueError("External and sampled communication fields conflict")
            context.update(sampled)
            for name, datum in context.items():
                append(frames, "context." + name, datum)
            for name, value in (("observations", obs), ("alive", active),
                                ("agent_reset", reset), ("available_actions", available),
                                ("states", state)):
                append(frames, name, value)
            if actor.output_kind == "logits" and critic is not None:
                # Off-policy targets reconstruct their own states; no behavior cache is needed.
                for name, value in _snapshot(actor_state).items():
                    append(frames, "actor_state." + name, value)
            for name, value in _snapshot(critic_state).items():
                append(frames, "critic_state." + name, value)
            team_valid = active.any().reshape(1).to(self.device)
            team_reset = torch.tensor([timestep == 0], device=self.device)
            if critic is not None:
                value = critic.step(state[None].to(self.device), critic_state, team_valid, team_reset)
                append(frames, "old_value", value.values[0])
            if episode_end:
                break  # No fictitious action or communication at the bootstrap frame.
            inputs = DecisionInput(obs[None].to(self.device), active[None].to(self.device),
                                   reset[None].to(self.device),
                                   {key: value[None].to(self.device) for key, value in context.items()})
            output = actor.step(inputs, actor_state)
            for name, datum in output.diagnostics.items():
                if name.startswith("record."):
                    append(transitions, "behavior." + name.removeprefix("record."), datum[0])
            log_prob = None
            if actor.output_kind == "logits":
                dist = masked_categorical(output.values[0], available.to(self.device), active.to(self.device))
                actions = (torch.multinomial(dist.probs.cpu(), 1, generator=rng).squeeze(-1)
                           if stochastic else dist.probs.argmax(-1).cpu())
                log_prob = dist.log_prob(actions.to(self.device)).cpu().masked_fill(~active, 0)
            else:
                feasible = masked_action_values(output.values[0], available.to(self.device), active.to(self.device))
                actions = feasible.argmax(-1).cpu()
                if stochastic and epsilon:
                    safe_available = available | ~active[:, None]
                    random_actions = torch.multinomial(safe_available.float(), 1, generator=rng).squeeze(-1)
                    explore = torch.rand(len(agents), generator=rng) < epsilon
                    actions = torch.where(explore, random_actions, actions)
                actions = actions.masked_fill(~active, 0)
            selected = {a: int(actions[i]) for i, a in enumerate(agents) if alive[a]}
            next_obs, reward, terms, truncs, infos = env.step(selected)
            for a in agents:
                dead[a] = bool(terms.get(a, dead[a]))
                # A reappearing slot is a new life, not a permanently dead unit.
                if a in env.agents and a in next_obs and not terms.get(a, False):
                    dead[a] = False
            terminal = all(dead.values())
            episode_end = (terminal or not env.agents or timestep + 1 >= self.max_steps
                           or all(dead[a] or truncs.get(a, False) for a in agents))
            survivor = {a: not dead[a] and (episode_end or a in env.agents) for a in agents}
            if not terminal and any(survivor[a] and a not in next_obs for a in agents):
                raise ValueError("Missing final observation; refusing a stale bootstrap")
            entries = {
                "actions": actions,
                "rewards": torch.tensor([reward.get(a, 0.) for a in agents], dtype=torch.float32),
                "agent_terminated": torch.tensor([dead[a] for a in agents]),
                "agent_truncated": torch.tensor([bool(truncs.get(a, False) or (episode_end and not terminal))
                                                  for a in agents]),
                "terminated": torch.tensor(terminal),
                "truncated": torch.tensor(episode_end and not terminal),
            }
            if log_prob is not None:
                entries["old_log_prob"] = log_prob
            for name, datum in entries.items():
                append(transitions, name, datum)
            previous_alive, alive = alive, survivor
            actor_state = output.state
            critic_state = value.state if critic is not None else {}
            observations, last_state = next_obs, state
        metrics = {}
        if callable(getattr(env, "episode_metrics", None)):
            metrics = {str(name): float(datum) for name, datum in env.episode_metrics().items()}
        elif isinstance(infos, dict):
            for name in ("battle_won", "success"):
                values = [float(info[name]) for info in infos.values() if isinstance(info, dict) and name in info]
                if values:
                    metrics[name] = sum(values) / len(values)
        if any(not np.isfinite(datum) for datum in metrics.values()):
            raise ValueError("Environment episode metrics must be finite scalars")
        return Episode(agents, {k: torch.stack(v) for k, v in frames.items()},
                       {k: torch.stack(v) for k, v in transitions.items()},
                       policy_version=policy_version, seed=seed,
                       metadata={"privileged_communication": bool(getattr(provider, "uses_global_state", False)),
                                 "metrics": metrics}).validate()
