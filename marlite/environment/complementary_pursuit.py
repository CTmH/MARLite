"""Pursuer-only complementary pursuit with explicit population sizes.

Native dynamics/rewards are unchanged. Observation filtering is applied here,
so every actor (including a raw teacher) receives the same legal input.
"""

import numpy as np
from pettingzoo.utils import BaseParallelWrapper
from magent2.environments.complementary_pursuit_v1 import parallel_env as NativeEnv


class PopulationEnv(NativeEnv):
    def __init__(self, n_pursuers=20, n_evaders=100, wall_density=0.04, **kwargs):
        for count in (n_pursuers, n_evaders):
            if type(count) is not int or count < 1:
                raise ValueError("Population sizes must be positive integers")
        if not 0 <= wall_density < 1:
            raise ValueError("wall_density must lie in [0,1)")
        map_size = kwargs.get("map_size", 45)
        if n_pursuers + n_evaders > (map_size - 2) ** 2 - int(
            map_size**2 * wall_density
        ):
            raise ValueError(
                "Requested population exceeds conservative free-cell capacity"
            )
        if n_pursuers + n_evaders > 1024:
            raise ValueError("Native 10-bit entity IDs require at most 1024 entities")
        self.n_pursuers, self.n_evaders = n_pursuers, n_evaders
        self.wall_density = wall_density
        super().__init__(**kwargs)

    def generate_map(self):
        evader, pursuer = self.env.get_handles()
        self.env.add_walls(method="random", n=int(self.map_size**2 * self.wall_density))
        self.env.add_agents(evader, method="random", n=self.n_evaders)
        self.env.add_agents(pursuer, method="random", n=self.n_pursuers)
        if (
            self.env.get_num(evader) != self.n_evaders
            or self.env.get_num(pursuer) != self.n_pursuers
        ):
            raise ValueError("Map cannot accommodate the requested population")

    def reset(self, seed=None, options=None):
        observations, infos = super().reset(seed=seed, options=options)
        # The native base caches the initial walls; resets generate a new map.
        self.base_state.fill(0)
        walls = self.env._get_walls_info()
        if len(walls):
            x, y = np.asarray(walls).T
            self.base_state[x, y, 0] = 1
        return observations, infos


class ComplementaryPursuit(BaseParallelWrapper):
    """Expose pursuers to MARLite; independently seeded random evaders."""

    def __init__(self, env, prey_policy="random"):
        super().__init__(env)
        if prey_policy not in ("random", "local_escape"):
            raise ValueError("Unknown fixed prey policy")
        self.prey_policy = prey_policy
        self.possible_agents = [
            a for a in env.possible_agents if a.startswith("pursuer_")
        ]
        self.agents = self.possible_agents[:]
        self.rng = np.random.default_rng()
        self.observation_spaces = {
            a: env.observation_space(a) for a in self.possible_agents
        }
        self.action_spaces = {a: env.action_space(a) for a in self.possible_agents}
        if any(space.n != 29 for space in self.action_spaces.values()):
            raise ValueError("Expected complementary-pursuit's 29-action schema")

    def _filter_observations(self, observations):
        result = {}
        for agent, observation in observations.items():
            if agent in self.action_spaces:
                observation = observation.copy()
                if self.env.minimap_mode:
                    observation[..., [3, 6]] = 0
                result[agent] = observation
        return result

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(np.random.SeedSequence([seed, 1741]))
        observations, infos = self.env.reset(seed=seed, options=options)
        self.agents = self.possible_agents[:]
        self.captured = 0
        self.effective_hits = self.ineffective_hits = self.missed_attacks = (
            self.attack_count
        ) = 0
        self._observations = observations
        self.initial_evaders = self.env.env.get_num(self.env.handles[0])
        return self._filter_observations(observations), {
            a: infos[a] for a in self.agents
        }

    def training_snapshot(self):
        """Privileged pre-step entity data for label validation; never an actor input."""
        native = self.env.env
        evader, pursuer = self.env.handles
        names = {index: name for name, index in self.env._agent_indices.items()}
        pursuers = {
            names[int(i)]: tuple(p)
            for i, p in zip(native.get_agent_id(pursuer), native.get_pos(pursuer))
        }
        normalized_hp, _ = native.get_state_observation(evader)
        initial_hp = self.env._ezpickle_kwargs["evader_hp"]
        evaders = {
            int(i): (*p, float(h) * initial_hp)
            for i, p, h in zip(
                native.get_agent_id(evader), native.get_pos(evader), normalized_hp
            )
        }
        return pursuers, evaders

    def episode_metrics(self):
        """Hit rates count target-events; miss rates count individual attack actions."""
        return {
            "capture_fraction": self.captured / self.initial_evaders,
            "effective_hit_rate": self.effective_hits
            / max(1, self.effective_hits + self.ineffective_hits),
            "missed_attack_rate": self.missed_attacks / max(1, self.attack_count),
        }

    def step(self, actions):
        if not self.agents:
            return {}, {}, {}, {}, {}
        if set(actions) - set(self.agents):
            raise ValueError("Actions include an inactive or non-pursuer agent")
        previous = self.agents[:]
        evaders = [a for a in self.env.agents if a.startswith("evader_")]
        complete = {
            a: int(self.rng.integers(self.env.action_space(a).n)) for a in evaders
        }
        if self.prey_policy == "local_escape":
            from marlite.environment.complementary_events import local_escape_action

            complete = {
                a: local_escape_action(
                    self._observations[a], self.rng, self.env.minimap_mode
                )
                for a in evaders
            }
        complete.update(actions)
        from marlite.environment.complementary_events import attack_outcomes

        pursuers, targets = self.training_snapshot()
        settings = self.env._ezpickle_kwargs
        events, missed = attack_outcomes(
            pursuers,
            targets,
            actions,
            settings["effective_damage"],
            settings["ineffective_damage"],
        )
        self.effective_hits += sum(e["effective"] for e in events.values())
        self.ineffective_hits += sum(not e["effective"] for e in events.values())
        self.missed_attacks += len(missed)
        self.attack_count += sum(action >= 5 for action in actions.values())
        obs, rewards, terms, truncs, infos = self.env.step(complete)
        remaining = self.env.env.get_num(self.env.handles[0])
        captures = len(evaders) - remaining
        # HP is exposed normalized; one-ULP reconstruction can straddle zero HP.
        definite = sum(
            e["captured"] and not e["capture_boundary"] for e in events.values()
        )
        possible = sum(e["captured"] or e["capture_boundary"] for e in events.values())
        if not definite <= captures <= possible:
            raise RuntimeError(
                "Attack label model disagrees with native capture events"
            )
        self.captured += captures
        self._observations = obs
        self.agents = [a for a in self.env.agents if a in self.action_spaces]
        for agent in previous:
            infos[agent] = dict(
                infos[agent],
                captured=self.captured,
                capture_fraction=self.captured / self.initial_evaders,
            )
        return (
            self._filter_observations(obs),
            {a: rewards[a] for a in previous},
            {a: terms[a] for a in previous},
            {a: truncs[a] for a in previous},
            {a: infos[a] for a in previous},
        )


def parallel_env(prey_policy="random", **kwargs):
    return ComplementaryPursuit(PopulationEnv(**kwargs), prey_policy=prey_policy)
