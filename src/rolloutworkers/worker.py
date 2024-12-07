import numpy as np
import torch
from pettingzoo import ParallelEnv
from ..environment.env_config import EnvConfig
from ..module.agents import AgentGroup
from ..module.model import RNNModel

class RolloutWorker:
    def __init__(self, env_config: EnvConfig, agent_group: AgentGroup, rnn_traj_len=5):
        self.env_config = env_config
        self.agent_group = agent_group
        self.rnn_traj_len = rnn_traj_len

        self.env = self.env_config.create_env()

    def generate_episode(self, episode_limit=None, epsilon=0.5):

        # Initialize the episode dictionary
        episode = {
            'observations':[],
            'state': [],
            'actions': [],
            'rewards': [],
            'avail_actions': [],
            'truncated': [],
            'episode_reward': 0,
            'win_tag': False,
            'episode_length': 0, 
        }

        win_tag = False
        episode_reward = 0

        for i in range(episode_limit+1):
            # Generate actions for each agent based on their observations and available actions
            # Collect the observations, actions, rewards, available actions, and truncations into the episode dictionary

            if i == 0:
                observations, infos = self.env.reset()
            else:
                episode['observations'].append(observations)
                episode['state'].append(self.env.state())
                episode['actions'].append(actions)
                episode['avail_actions'].append(avail_actions)

                observations, rewards, terminations, truncations, infos = self.env.step(actions)
                
                episode['rewards'].append(rewards)
                episode['truncated'].append(truncations)

                episode_reward += np.sum(np.array([rewards[agent] for agent in rewards.keys()]))

                # TODO Need to add win tag logic here
                if True in terminations.values():
                    # Append last state and action before termination
                    episode['observations'].append(observations)
                    episode['state'].append(self.env.state())
                    # Padding actions and avail_actions with None to match the trajectory length
                    episode['actions'].append(None)
                    episode['rewards'].append(None)
                    episode['avail_actions'].append(None)
                    episode['truncated'].append(None)
                    break

            avail_actions = {agent: self.env.action_space(agent) for agent in self.env.agents}
            processed_obs = self._obs_preprocess([observations], self.agent_group)
            actions = self.agent_group.act(processed_obs, avail_actions, epsilon)
            
        episode['episode_length'] = len(episode['observations'])
        episode['episode_reward'] = episode_reward
        episode['win_tag'] = win_tag

        # Close the environment after generating the episode
        self.env.close()

        return episode
    
    def _obs_preprocess(self, observations: list, agent_group: AgentGroup):
        agents = self.agent_group.agents
        models = self.agent_group.models
        processed_obs = {agent_id : [] for agent_id in agents.keys()}
        for agent_id, model_name in agents.items():
            if isinstance(models[model_name], RNNModel):
                obs = [o[agent_id] for o in observations[-self.rnn_traj_len:]]
            else:
                obs = observations[-1][agent_id]
            processed_obs[agent_id] = np.array(obs)
        return processed_obs