import numpy as np
from copy import deepcopy
from ..environment.env_config import EnvConfig
from ..algorithm.agents import AgentGroup
from ..algorithm.agents.graph_agent_group import GraphAgentGroup
from ..algorithm.model import TimeSeqModel

def _obs_preprocess(observations: list, agent_model_dict: dict, models: dict, rnn_traj_len: int):
        agents = agent_model_dict.keys()
        processed_obs = {agent_id : [] for agent_id in agents}
        for agent_id, model_name in agent_model_dict.items():
            if isinstance(models[model_name], TimeSeqModel):
                obs_len = len(observations)
                if obs_len < rnn_traj_len:
                    padding_length = rnn_traj_len - obs_len
                    obs_padding = [np.zeros_like(observations[-1][agent_id]) for _ in range(padding_length)]
                    obs = obs_padding + [o[agent_id] for o in observations[-rnn_traj_len:]]
                else:
                    obs = [o[agent_id] for o in observations[-rnn_traj_len:]]
            else:
                obs = [observations[-1].get(agent_id)]
            processed_obs[agent_id] = np.array(obs)
        return processed_obs

def _ensure_all_agents_present(data_dict: dict, default_values: dict) -> dict:
    """
    Ensure that the dictionary contains all possible agents.
    If any agent is missing, add it with the corresponding default value.
    Maintains the order of possible_agents.
    """

    result = {}
    for agent in default_values.keys():
        if agent in data_dict:
            result[agent] = data_dict[agent]
        elif agent in default_values:
            result[agent] = default_values[agent]
        else:
            # Fallback: use first available value as template
            if data_dict:
                first_value = next(iter(data_dict.values()))
                result[agent] = np.zeros_like(first_value)
            elif default_values:
                first_default = next(iter(default_values.values()))
                result[agent] = np.zeros_like(first_default)
    return result

def multiprocess_rollout(env_config: EnvConfig,
            agent_group: AgentGroup,
            rnn_traj_len=5,
            episode_limit=100,
            epsilon=0.5,
            device='cpu'):
    """Execute a rollout using multiprocess environment.
    
    Args:
        env_config: Environment configuration
        agent_group: Agent group for acting
        rnn_traj_len: Trajectory length for RNN models
        episode_limit: Maximum steps per episode
        epsilon: Exploration rate
        device: Device to run the model on
    
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
    possible_agents = None
    default_observations = {}
    default_avail_actions = {}
    use_action_mask = False

    for i in range(episode_limit + 1):
         # Reset environment
        if i == 0:
            observations, infos = env.reset()

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
            episode['observations'].append(observations)
            episode['states'].append(env.state())
            episode['edge_indices'].append(edge_indices)
            episode['actions'].append(actions)
            episode['avail_actions'].append(avail_actions)

            # Step environment
            observations, rewards, terminations, truncations, infos = env.step(actions)

            # Store post-step data
            episode['rewards'].append(rewards)
            episode['truncated'].append(truncations)
            episode['terminations'].append(terminations)
            episode['next_states'].append(env.state())
            episode['next_observations'].append(observations)
            
            # Update episode reward
            all_agents_rewards = [value for _, value in rewards.items()]
            episode['all_agents_sum_rewards'].append(sum(all_agents_rewards))
            episode_reward += np.sum(np.array([rewards[agent] for agent in rewards.keys()]))

            # TODO win tag logic here
            # TODO logic for lost units
            # TODO reward of the last state and the second last state
            # Check termination conditions
            if True in terminations.values() or True in truncations.values():
                break

        # Ensure all possible agents are present in observations and available actions
        observations = _ensure_all_agents_present(observations, default_observations)

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
        avail_actions = _ensure_all_agents_present(current_avail_actions, default_avail_actions)

        # Get actions from agent
        processed_obs = _obs_preprocess(
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

    episode['episode_length'] = len(episode['observations'])
    episode['episode_reward'] = episode_reward
    episode['win_tag'] = win_tag

    env.close()
    return episode
