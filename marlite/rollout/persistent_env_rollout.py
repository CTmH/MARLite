import signal
import numpy as np
from copy import deepcopy
from typing import Callable, List, Dict, Any
import time
from marlite.environment import EnvConfig
from marlite.algorithm.agents import AgentGroup, GraphAgentGroup
from marlite.util.env_util import obs_preprocess, ensure_all_agents_present

def persistent_env_rollout(env_config: EnvConfig,
            agent_group: AgentGroup,
            n_episodes: int,
            rnn_traj_len=5,
            episode_limit=100,
            epsilon=0.5,
            device='cpu',
            check_victory: Callable = None) -> List[Dict[str, Any]]:
    """Execute multiple rollouts using a single environment instance.

    Args:
        env_config: Environment configuration
        agent_group: Agent group for acting
        n_episodes: Number of episodes to run
        rnn_traj_len: Trajectory length for RNN models
        episode_limit: Maximum steps per episode
        epsilon: Exploration rate
        device: Device to run the model on
        check_victory: Optional function that takes (env, infos) and returns whether the game was won

    Returns:
        List of episode data dictionaries
    """

    # Setup environment and agent
    env = None
    for attempt in range(3):  # Retry up to 3 times
        try:
            env = env_config.create_env()
            break
        except Exception as e:
            print(f"Create environment failed with error: {e}")
            env = None
            time.sleep(0.5)
            continue

    if env is None:
        print("Environment creation failed after 3 attempts")
        return []

    agent_group = deepcopy(agent_group).reset().eval().to(device)

    episodes = []

    for episode_idx in range(n_episodes):
        # Initialize episode data
        episode = {
            'alive_mask' : [],
            'observations': [],
            'states': [],
            'edge_indices': [],
            'actions': [],
            'rewards': [],
            'avail_actions': [],
            'truncations': [],
            'terminations': [],
            'next_states': [],
            'next_observations': [],
            'next_avail_actions': [],
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
        default_alive_mask = {agent: False for agent in env.possible_agents}
        default_rewards = {agent: 0 for agent in env.possible_agents}
        default_terminations = {agent: True for agent in env.possible_agents}
        default_truncations = {agent: True for agent in env.possible_agents}
        use_action_mask = False

        for i in range(episode_limit + 1):
            # Reset environment
            if i == 0:
                # Generate random seed based on current time
                seed = int(time.time() * 1000) % (2**32 - 1)

                # Reset environment
                try:
                    observations, infos = env.reset(seed=seed)
                except Exception as e:
                    print(f"Reset failed: {e}")
                    # Close environment and return collected episodes
                    #env.close()
                    return episodes  # Return any collected episodes

                # Determine if action masking is used
                info_item = next(iter(infos.values()), None)
                if isinstance(info_item, dict) and isinstance(info_item.get('action_mask'), np.ndarray):
                    use_action_mask = True

                # Create default observations for each possible agent
                for agent in env.possible_agents:
                    if agent in observations:
                        default_observations[agent] = np.zeros_like(observations[agent])
                    else:
                        # If agent not present in initial observations, use first available observation as template
                        first_obs = next(iter(observations.values()))
                        default_observations[agent] = np.zeros_like(first_obs)

                # Create default available actions for each possible agent
                for agent in env.possible_agents:
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
                episode['alive_mask'].append(alive_mask)
                episode['observations'].append(observations)
                episode['states'].append(env.state())
                episode['edge_indices'].append(edge_indices)
                episode['actions'].append(all_actions)
                episode['avail_actions'].append(avail_actions)

                # Step environment
                try:
                    observations, rewards, terminations, truncations, infos = env.step(actions)
                except Exception as e:
                    print(f"Step failed: {e}")
                    # Remove last added items
                    for key in ['alive_mask', 'observations', 'states', 'edge_indices', 'actions', 'avail_actions']:
                        if episode[key]:
                            episode[key].pop()
                    break  # Skip this episode if step fails

                # Ensure all possible agents are present in observations and rewards
                observations = ensure_all_agents_present(observations, default_observations)
                rewards = ensure_all_agents_present(rewards, default_rewards)
                terminations = ensure_all_agents_present(terminations, default_terminations)
                truncations = ensure_all_agents_present(truncations, default_truncations)

                # Store post-step data
                episode['rewards'].append(rewards)
                episode['truncations'].append(truncations)
                episode['terminations'].append(terminations)
                episode['next_states'].append(env.state())
                episode['next_observations'].append(observations)

                # Update episode reward
                agent_reward_sum = sum(rewards.values())
                episode['all_agents_sum_rewards'].append(agent_reward_sum)
                episode_reward += agent_reward_sum

                # Check if the game was won using the provided function
                if check_victory is not None:
                    win_tag = check_victory(env, infos)
                if win_tag or not env.agents:
                    episode['next_avail_actions'].append(default_avail_actions)
                    break
                #if not env.agents:  # Game has ended
                #    episode['next_avail_actions'].append(default_avail_actions)
                #    break

            # Update Alive agent mask
            alive_mask = ensure_all_agents_present({agent: True for agent in env.agents}, default_alive_mask)

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
            if i > 0:
                episode['next_avail_actions'].append(avail_actions)

            # Get actions from agent
            processed_obs, traj_padding_mask = obs_preprocess(
                observations=episode['observations'] + [observations],
                agents=agent_group.agent_model_dict.keys(),
                rnn_traj_len=rnn_traj_len
            )

            ret = agent_group.act(processed_obs, env.state(), avail_actions, traj_padding_mask, env.agents, epsilon)
            actions, all_actions = ret['actions'], ret['all_actions']
            edge_indices = ret.get('edge_indices', None)

        # Check if the game was won using the provided function
        #if check_victory is not None:
        #    win_tag = check_victory(env, infos)
        episode['win_tag'] = win_tag
        episode['episode_length'] = len(episode['observations'])
        episode['episode_reward'] = episode_reward

        # Only add episode if it has observations (i.e., was successfully executed)
        if 'observations' in episode and len(episode['observations']) > 0:
            episodes.append(episode)

    env.close()
    return episodes