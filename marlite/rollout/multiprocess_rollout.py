"""Single-episode rollout worker for multiprocess environments.

Collects a single episode using a pluggable set of collection phases.
Pass ``required_attrs`` (``None`` = full profile, ``str`` = named profile)
to control which attributes are stored in the episode dictionary.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import time
from multiprocessing.shared_memory import SharedMemory
from marlite.environment.env_config import EnvConfig
from marlite.algorithm.agents import AgentGroup, AgentGroupConfig
from marlite.util.env_util import obs_preprocess, ensure_all_agents_present
from marlite.util.serialization import deserialize_from_buffer
from marlite.rollout.attribute_spec import (
    resolve_required_attrs,
    get_timestep_attrs,
)
from marlite.rollout.phases import resolve_phases, RolloutPhases


def multiprocess_rollout(
    env_config: EnvConfig,
    agent_group_config: AgentGroupConfig,
    shm_info: Tuple[str, int],
    rnn_traj_len: int = 5,
    episode_limit: int = 100,
    epsilon: float = 0.5,
    device: str = "cpu",
    check_victory: Optional[Callable] = None,
    required_attrs: Optional[Union[str, List[str], tuple]] = None,
):
    """Execute a rollout using multiprocess environment.

    Parameters
    ----------
    env_config : EnvConfig
        Environment configuration.
    agent_group_config : AgentGroupConfig
        Agent group configuration.
    shm_info : Tuple[str, int]
        (shared_memory_name, size) for serialized agent group params.
    rnn_traj_len : int
        Trajectory length for RNN models.
    episode_limit : int
        Maximum steps per episode.
    epsilon : float
        Exploration rate.
    device : str
        Device to run the model on.
    check_victory : Callable or None
        Optional function (env, infos) -> bool indicating a win.
    required_attrs : str or None
        Profile name selecting which attributes to collect.
        ``None`` or ``"full"`` collects everything (backward compatible).

    Returns
    -------
    episode : dict or None
        Episode data dictionary, or ``None`` if the episode could not be
        collected.
    """
    # ---- Deserialize agent group from shared memory ----
    shm_name, shm_size = shm_info
    shm = SharedMemory(name=shm_name)
    serialized_bytes = bytes(shm.buf[:shm_size])
    shm.close()

    agent_group = agent_group_config.get_agent_group()
    agent_group.load_state_dict(deserialize_from_buffer(serialized_bytes))

    env = env_config.create_env()
    agent_group = agent_group.reset().eval().to(device)
    possible_agents = env.possible_agents.copy()

    # ---- Resolve required attributes and collection phases ----
    attrs_list = resolve_required_attrs(required_attrs)
    phases = resolve_phases(required_attrs)
    timestep_attr_names = get_timestep_attrs(attrs_list)

    # Pre-initialize episode dict with empty lists for every timestep attr
    episode: Dict[str, Any] = {attr: [] for attr in timestep_attr_names}
    episode["episode_reward"] = 0
    episode["win_tag"] = False
    episode["episode_length"] = 0

    win_tag = False
    episode_reward = 0

    # Default values for dead / missing agents
    default_observations: Dict[Any, Any] = {}
    default_avail_actions: Dict[Any, Any] = {}
    default_alive_mask = {agent: False for agent in possible_agents}
    default_rewards = {agent: 0 for agent in possible_agents}
    default_terminations = {agent: True for agent in possible_agents}
    default_truncations = {agent: True for agent in possible_agents}
    default_all_log_probs = {agent: 0.0 for agent in possible_agents}
    default_log_probs = {agent: 0.0 for agent in possible_agents}
    use_action_mask = False

    # Variables set in the loop — initialise to None to satisfy the type checker
    alive_mask: Any = None
    observations: Any = None
    actions: Any = None
    all_actions: Any = None
    edge_indices: Any = None
    all_group_indices: Any = None
    all_log_probs: Any = None
    log_probs: Any = None
    avail_actions: Any = None
    infos: Any = None

    for i in range(episode_limit + 1):
        # ===========================================================
        # [RESET]  (i == 0) — fixed logic
        # ===========================================================
        if i == 0:
            seed = int(time.time() * 1000) % (2**24 - 1)

            try:
                observations, infos = env.reset(seed=seed)
            except Exception as e:
                print("Reset failed")
                return None

            info_item = next(iter(infos.values()), None)
            if isinstance(info_item, dict) and isinstance(
                info_item.get("action_mask"), np.ndarray
            ):
                use_action_mask = True

            for agent in possible_agents:
                if agent in observations:
                    default_observations[agent] = np.zeros_like(observations[agent])
                else:
                    first_obs = next(iter(observations.values()))
                    default_observations[agent] = np.zeros_like(first_obs)

            for agent in possible_agents:
                if use_action_mask:
                    if agent in infos and "action_mask" in infos[agent]:
                        default_avail_actions[agent] = np.ones_like(
                            infos[agent]["action_mask"], dtype=np.int8
                        )
                    else:
                        first_mask = next(iter(infos.values()))["action_mask"]
                        default_avail_actions[agent] = np.ones_like(
                            first_mask, dtype=np.int8
                        )
                else:
                    if agent in env.agents:
                        default_avail_actions[agent] = env.action_space(agent)
                    else:
                        default_avail_actions[agent] = env.action_space(
                            next(iter(env.agents))
                        )

        # ===========================================================
        # [PRE_STEP + ENV_STEP + POST_STEP + TERMINAL]  (i > 0)
        # ===========================================================
        else:
            # -------------------------------------------------------
            # [PRE_STEP] — observation-phase attrs
            # -------------------------------------------------------
            ctx_pre = {
                "alive_mask": alive_mask,
                "observations": observations,
                "states": env.state(),
                "edge_indices": edge_indices,
                "group_indices": all_group_indices,
                "actions": all_actions,
                "all_log_probs": all_log_probs,
                "log_probs": log_probs,
                "avail_actions": avail_actions,
                "infos": infos,
            }
            # Record lengths before pre_step for error-rollback
            lengths_before = {
                k: len(v) for k, v in episode.items() if isinstance(v, list)
            }
            phases.pre_step(episode, ctx_pre)

            # -------------------------------------------------------
            # [ENV_STEP] — fixed logic with error rollback
            # -------------------------------------------------------
            try:
                observations, rewards, terminations, truncations, infos = env.step(
                    actions
                )
            except Exception as e:
                print(f"Step failed: {e}")
                _rollback(episode, lengths_before)
                break

            observations = ensure_all_agents_present(
                observations, default_observations
            )
            rewards = ensure_all_agents_present(rewards, default_rewards)
            terminations = ensure_all_agents_present(
                terminations, default_terminations
            )
            truncations = ensure_all_agents_present(
                truncations, default_truncations
            )

            # -------------------------------------------------------
            # [POST_STEP] — env.step() result attrs
            # -------------------------------------------------------
            ctx_post = {
                "rewards": rewards,
                "terminations": terminations,
                "truncations": truncations,
                "observations": observations,   # these are next_observations
                "all_agents_sum_rewards": sum(rewards.values()),
            }
            phases.post_step(episode, ctx_post)

            agent_reward_sum = sum(rewards.values())
            episode_reward += agent_reward_sum

            # -------------------------------------------------------
            # [TERMINAL] — next_states (always) + terminal phase
            # -------------------------------------------------------
            if check_victory is not None:
                win_tag = check_victory(env, infos)
            if win_tag or not env.agents:
                # Reuse last state — some envs raise on env.state()
                # after termination.
                episode["next_states"].append(episode["states"][-1])
                ctx_term = {
                    "default_avail_actions": default_avail_actions,
                    "default_alive_mask": default_alive_mask,
                    "edge_indices": edge_indices,
                    "group_indices": all_group_indices,
                }
                phases.terminal(episode, ctx_term)
                break
            episode["next_states"].append(env.state())

        # ===========================================================
        # [COMPUTE_OBSERVE + COMPUTE_ACT] — fixed logic (every i)
        # ===========================================================
        alive_mask = ensure_all_agents_present(
            {agent: True for agent in env.agents}, default_alive_mask
        )

        if use_action_mask:
            current_avail_actions = {}
            for agent in env.agents:
                if agent in infos and "action_mask" in infos[agent]:
                    current_avail_actions[agent] = np.array(
                        infos[agent]["action_mask"], dtype=np.int8
                    )
        else:
            current_avail_actions = {
                agent: env.action_space(agent) for agent in env.agents
            }
        avail_actions = ensure_all_agents_present(
            current_avail_actions, default_avail_actions
        )

        processed_obs, traj_padding_mask = obs_preprocess(
            observations=episode["observations"] + [observations],
            agents=agent_group.agent_model_dict.keys(),
            rnn_traj_len=rnn_traj_len,
        )

        ret = agent_group.act(
            processed_obs,
            env.state(),
            avail_actions,
            traj_padding_mask,
            env.agents,
            epsilon,
        )
        actions, all_actions = ret["actions"], ret["all_actions"]
        edge_indices = ret.get("edge_indices", np.zeros((2, 0)))
        all_group_indices = ret.get(
            "all_group_indices", {agent: -1 for agent in possible_agents}
        )
        all_log_probs = ret.get("all_log_probs", default_all_log_probs)
        log_probs = ret.get("log_probs", default_log_probs)

        # ===========================================================
        # [NEXT] — next-* attrs after agent.act()  (i > 0)
        # ===========================================================
        if i > 0:
            ctx_next = {
                "alive_mask": alive_mask,
                "avail_actions": avail_actions,
                "edge_indices": edge_indices,
                "group_indices": all_group_indices,
            }
            phases.next_attr(episode, ctx_next)

    # ===============================================================
    # [FINALIZE] — episode-level attrs
    # ===============================================================
    ctx_final = {
        "win_tag": win_tag,
        "episode_length": len(episode.get("observations", [])),
        "episode_reward": episode_reward,
    }
    phases.finalize(episode, ctx_final)

    env.close()
    return episode


def _rollback(episode: dict, lengths_before: dict) -> None:
    """Truncate episode lists back to their lengths before the pre_step phase."""
    for key, length in lengths_before.items():
        if isinstance(episode[key], list):
            episode[key] = episode[key][:length]
