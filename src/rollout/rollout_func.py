import numpy as np
from copy import deepcopy
from typing import Callable
import time
from ..environment.env_config import EnvConfig
from ..algorithm.agents import AgentGroup
from ..algorithm.agents.graph_agent_group import GraphAgentGroup
from src.util.env_util import obs_preprocess, ensure_all_agents_present

def multiprocess_rollout(env_config: EnvConfig,
            agent_group: AgentGroup,
            rnn_traj_len=5,
            episode_limit=100,
            epsilon=0.5,
            device='cpu',
            check_victory: Callable = None):
    """Execute a rollout using multiprocess environment.

    Args:
        env_config: Environment configuration
        agent_group: Agent group for acting
        rnn_traj_len: Trajectory length for RNN models
        episode_limit: Maximum steps per episode
        epsilon: Exploration rate
        device: Device to run the model on
        check_victory: Optional function that takes (env, infos) and returns whether the game was won

    Returns:
        Episode data dictionary
    """

     # Setup environment and agent
    env = env_config.create_env()
    agent_group = deepcopy(agent_group).reset().eval().to(device)
    possible_agents = env.possible_agents.copy()

    # Initialize episode data
    episode = {
        'observations': [],
        'states': [],
        'edge_indices': [],
        'actions': [],
        'rewards': [],
        'avail_actions': [],
        'truncated': [],
        'terminations': [],
        'next_states': [],
        'next_observations': [],
        'all_agents_sum_rewards': [],
        'episode_reward': 0,
        'win_tag': False,
        'episode_length': 0,
    }

    # Initialize tracking variables
    win_tag = False
    episode_reward = 0

    # Initialize variables for default observations and available actions
    default_observations = {}
    default_avail_actions = {}
    default_rewards = {agent: 0 for agent in possible_agents}
    default_terminations = {agent: True for agent in possible_agents}
    default_truncations = {agent: True for agent in possible_agents}
    use_action_mask = False

    for i in range(episode_limit + 1):
         # Reset environment
        if i == 0:
            # Generate random seed based on current time
            seed = int(time.time() * 1000) % (2**32 - 1)

            # Reset environment with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    observations, infos = env.reset(seed=seed)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        # TODO Need log support
                        return None
                    # Recreate environment and try again
                    env.close()
                    env = env_config.create_env()
                    print(f"Reset failed, attempt {attempt + 1}/{max_retries}: {e}")

            # Determine if action masking is used
            info_item = next(iter(infos.values()), None)
            if isinstance(info_item, dict) and isinstance(info_item.get('action_mask'), np.ndarray):
                use_action_mask = True

            # Create default observations for each possible agent
            for agent in possible_agents:
                if agent in observations:
                    default_observations[agent] = np.zeros_like(observations[agent])
                else:
                    # If agent not present in initial observations, use first available observation as template
                    first_obs = next(iter(observations.values()))
                    default_observations[agent] = np.zeros_like(first_obs)

            # Create default available actions for each possible agent
            for agent in possible_agents:
                if use_action_mask:
                    if agent in infos and 'action_mask' in infos[agent]:
                        default_avail_actions[agent] = np.ones_like(infos[agent]['action_mask'], dtype=np.int8)
                    else:
                        # Use first available action mask as template
                        first_mask = next(iter(infos.values()))['action_mask']
                        default_avail_actions[agent] = np.ones_like(first_mask, dtype=np.int8)
                else:
                    if agent in env.agents:
                        default_avail_actions[agent] = env.action_space(agent)
                    else:
                        # Use first available action space as template
                        default_avail_actions[agent] = env.action_space(next(iter(env.agents)))

        # Step environment
        else:
            # Store transition data
            episode['observations'].append(observations)
            episode['states'].append(env.state())
            episode['edge_indices'].append(edge_indices)
            episode['actions'].append(actions)
            episode['avail_actions'].append(avail_actions)

            # Step environment
            actual_actions = {agent:actions[agent] for agent in env.agents}
            observations, rewards, terminations, truncations, infos = env.step(actual_actions)

            # Ensure all possible agents are present in observations and rewards
            observations = ensure_all_agents_present(observations, default_observations)
            rewards = ensure_all_agents_present(rewards, default_rewards)
            terminations = ensure_all_agents_present(terminations, default_terminations)
            truncations = ensure_all_agents_present(truncations, default_truncations)

            # Store post-step data
            episode['rewards'].append(rewards)
            episode['truncated'].append(truncations)
            episode['terminations'].append(terminations)
            episode['next_states'].append(env.state())
            episode['next_observations'].append(observations)

            # Update episode reward
            agent_reward_sum = sum(rewards.values())
            episode['all_agents_sum_rewards'].append(agent_reward_sum)
            episode_reward += agent_reward_sum

            if not env.agents:  # Game has ended
                break

        # Create available actions dictionary
        if use_action_mask:
            # Create available actions from action masks
            current_avail_actions = {}
            for agent in env.agents:
                if agent in infos and 'action_mask' in infos[agent]:
                    current_avail_actions[agent] = np.array(infos[agent]['action_mask'], dtype=np.int8)
        else:
            # Create available actions from action spaces
            current_avail_actions = {agent: env.action_space(agent) for agent in env.agents}
        avail_actions = ensure_all_agents_present(current_avail_actions, default_avail_actions)

        # Get actions from agent
        processed_obs = obs_preprocess(
            observations=episode['observations'] + [observations],
            agent_model_dict=agent_group.agent_model_dict,
            models=agent_group.models,
            rnn_traj_len=rnn_traj_len
        )
        if isinstance(agent_group, GraphAgentGroup):
            ret = agent_group.act(processed_obs, env.state(), avail_actions, epsilon)
        else:
            ret = agent_group.act(processed_obs, avail_actions, epsilon)
        actions = ret['actions']
        edge_indices = ret.get('edge_indices', None)

    # TODO win tag logic here
    # TODO logic for lost units
    # TODO reward of the last state and the second last state

    # Check if the game was won using the provided function
    if check_victory is not None:
        win_tag = check_victory(env, infos)
    episode['win_tag'] = win_tag
    episode['episode_length'] = len(episode['observations'])
    episode['episode_reward'] = episode_reward

    env.close()
    return episode