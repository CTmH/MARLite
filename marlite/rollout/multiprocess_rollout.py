import numpy as np
from typing import Callable, Tuple
import time
from multiprocessing.shared_memory import SharedMemory
from marlite.environment.env_config import EnvConfig
from marlite.algorithm.agents import AgentGroup, AgentGroupConfig
from marlite.util.env_util import obs_preprocess, ensure_all_agents_present
from marlite.util.serialization import deserialize_from_buffer


def multiprocess_rollout(
    env_config: EnvConfig,
    agent_group_config: AgentGroupConfig,
    shm_info: Tuple[str, int],
    rnn_traj_len=5,
    episode_limit=100,
    epsilon=0.5,
    device="cpu",
    check_victory: Callable = None,
):
    """Execute a rollout using multiprocess environment.

    Args:
        env_config: Environment configuration
        agent_group_config: AgentGroupConfig instance
        shm_info: Tuple of (shared_memory_name, size) for serialized agent group params
        rnn_traj_len: Trajectory length for RNN models
        episode_limit: Maximum steps per episode
        epsilon: Exploration rate
        device: Device to run the model on
        check_victory: Optional function that takes (env, infos) and returns whether the game was won

    Returns:
        Episode data dictionary
    """
    shm_name, shm_size = shm_info
    shm = SharedMemory(name=shm_name)
    serialized_bytes = bytes(shm.buf[:shm_size])
    shm.close()

    agent_group = agent_group_config.get_agent_group()
    agent_group.load_state_dict(deserialize_from_buffer(serialized_bytes))

    env = env_config.create_env()
    agent_group = agent_group.reset().eval().to(device)
    possible_agents = env.possible_agents.copy()

    episode = {
        "alive_mask": [],
        "observations": [],
        "states": [],
        "edge_indices": [],
        "zone_indices": [],
        "actions": [],
        "rewards": [],
        "avail_actions": [],
        "truncations": [],
        "terminations": [],
        "next_alive_mask": [],
        "next_edge_indices": [],
        "next_zone_indices": [],
        "next_states": [],
        "next_observations": [],
        "next_avail_actions": [],
        "infos": [],
        "all_agents_sum_rewards": [],
        "episode_reward": 0,
        "win_tag": False,
        "episode_length": 0,
    }

    win_tag = False
    episode_reward = 0

    default_observations = {}
    default_avail_actions = {}
    default_alive_mask = {agent: False for agent in possible_agents}
    default_rewards = {agent: 0 for agent in possible_agents}
    default_terminations = {agent: True for agent in possible_agents}
    default_truncations = {agent: True for agent in possible_agents}
    use_action_mask = False

    for i in range(episode_limit + 1):
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

        else:
            episode["alive_mask"].append(alive_mask)
            episode["observations"].append(observations)
            episode["states"].append(env.state())
            episode["edge_indices"].append(edge_indices)
            episode["zone_indices"].append(zone_indices)
            episode["actions"].append(all_actions)
            episode["avail_actions"].append(avail_actions)
            episode["infos"].append(infos)
            try:
                observations, rewards, terminations, truncations, infos = env.step(
                    actions
                )
            except Exception as e:
                print(f"Step failed, Return None")
                return None

            observations = ensure_all_agents_present(observations, default_observations)
            rewards = ensure_all_agents_present(rewards, default_rewards)
            terminations = ensure_all_agents_present(terminations, default_terminations)
            truncations = ensure_all_agents_present(truncations, default_truncations)

            episode["rewards"].append(rewards)
            episode["truncations"].append(truncations)
            episode["terminations"].append(terminations)
            episode["next_observations"].append(observations)

            agent_reward_sum = sum(rewards.values())
            episode["all_agents_sum_rewards"].append(agent_reward_sum)
            episode_reward += agent_reward_sum

            if check_victory is not None:
                win_tag = check_victory(env, infos)
            if win_tag or not env.agents:
                episode["next_states"].append(episode["states"][-1])
                episode["next_avail_actions"].append(default_avail_actions)
                episode["next_alive_mask"].append(default_alive_mask)
                episode["next_edge_indices"].append(edge_indices)
                episode["next_zone_indices"].append(zone_indices)
                break
            episode["next_states"].append(env.state())

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
        zone_indices = ret.get("zone_indices", np.zeros(len(possible_agents), dtype=np.int8))

        if i > 0:
            episode["next_alive_mask"].append(alive_mask)
            episode["next_avail_actions"].append(avail_actions)
            episode["next_edge_indices"].append(edge_indices)
            episode["next_zone_indices"].append(zone_indices)

    episode["win_tag"] = win_tag
    episode["episode_length"] = len(episode["observations"])
    episode["episode_reward"] = episode_reward

    env.close()
    return episode
