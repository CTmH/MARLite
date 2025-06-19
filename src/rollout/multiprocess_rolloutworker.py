import torch.multiprocessing as mp
import numpy as np
import logging
from copy import deepcopy
from ..environment.env_config import EnvConfig
from ..algorithm.agents import AgentGroup
from ..algorithm.agents.graph_agent_group import GraphAgentGroup
from ..algorithm.model import TimeSeqModel

class MultiProcessRolloutWorker(mp.Process):
    def __init__(self,
                 env_config: EnvConfig,
                 agent_group: AgentGroup,
                 rnn_traj_len=5,
                 episode_limit=100,
                 epsilon=0.5,
                 device='cpu'):
        super(MultiProcessRolloutWorker, self).__init__()

        # 共享参数需要深拷贝
        self.env_config = deepcopy(env_config)
        self.agent_group = deepcopy(agent_group)
        self.rnn_traj_len = rnn_traj_len
        self.episode_limit = episode_limit
        self.epsilon = epsilon
        self.device = device

    def run(self):
        self.agent_group = deepcopy(self.agent_group).to(self.device)
        return self.rollout()

    def rollout(self):
        env = self.env_config.create_env()
        agent_group = deepcopy(self.agent_group).eval().to(self.device)

        # Initialize the episode dictionary
        episode = {
            'observations':[],
            'states': [],
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

        win_tag = False
        episode_reward = 0

        for i in range(self.episode_limit+1):
            # Generate actions for each agent based on their observations and available actions
            # Collect the observations, actions, rewards, available actions, and truncations into the episode dictionary

            if i == 0:
                observations, infos = env.reset()
            else:
                episode['observations'].append(observations)
                episode['states'].append(env.state())
                episode['actions'].append(actions)
                episode['avail_actions'].append(avail_actions)

                observations, rewards, terminations, truncations, infos = env.step(actions)

                episode['rewards'].append(rewards)
                all_agents_rewards = [value for _, value in rewards.items()]
                episode['all_agents_sum_rewards'].append(sum(all_agents_rewards))
                episode['truncated'].append(truncations)
                episode['terminations'].append(terminations)
                episode['next_states'].append(env.state()) # TODO: Check if this is correct
                episode['next_observations'].append(observations) # TODO: Check if this is correct

                episode_reward += np.sum(np.array([rewards[agent] for agent in rewards.keys()]))

                # TODO win tag logic here
                # TODO logic for lost units
                # TODO reward of the last state and the second last state
                if True in terminations.values() or True in truncations.values():
                    break

            avail_actions = {agent: env.action_space(agent) for agent in env.agents}
            processed_obs = _obs_preprocess(
                observations=episode['observations']+[observations],
                agent_model_dict=agent_group.agent_model_dict,
                models=agent_group.models,
                rnn_traj_len=self.rnn_traj_len)
            if isinstance(agent_group, GraphAgentGroup):
                actions = agent_group.act(processed_obs, env.state(), avail_actions, self.epsilon)
            else:
                actions = agent_group.act(processed_obs, avail_actions, self.epsilon)

        episode['episode_length'] = len(episode['observations'])
        episode['episode_reward'] = episode_reward
        episode['win_tag'] = win_tag

        # Close the environment after generating the episode
        env.close()

        return episode

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

def rollout(env_config: EnvConfig,
            agent_group: AgentGroup,
            rnn_traj_len=5,
            episode_limit=100,
            epsilon=0.5,
            device='cpu'):
    env = env_config.create_env()
    agent_group = deepcopy(agent_group).reset().eval().to(device)

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

    win_tag = False
    episode_reward = 0

    for i in range(episode_limit + 1):
        if i == 0:
            observations, infos = env.reset()
        else:
            episode['observations'].append(observations)
            episode['states'].append(env.state())
            episode['edge_indices'].append(edge_indices)
            episode['actions'].append(actions)
            episode['avail_actions'].append(avail_actions)

            observations, rewards, terminations, truncations, infos = env.step(actions)

            episode['rewards'].append(rewards)
            all_agents_rewards = [value for _, value in rewards.items()]
            episode['all_agents_sum_rewards'].append(sum(all_agents_rewards))
            episode['truncated'].append(truncations)
            episode['terminations'].append(terminations)
            episode['next_states'].append(env.state())
            episode['next_observations'].append(observations)

            episode_reward += np.sum(np.array([rewards[agent] for agent in rewards.keys()]))

            if True in terminations.values() or True in truncations.values():
                break

        avail_actions = {agent: env.action_space(agent) for agent in env.agents}
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
