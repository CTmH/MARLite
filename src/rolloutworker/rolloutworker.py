import numpy as np
import torch
from pettingzoo import ParallelEnv
from ..environment.env_config import EnvConfig
from ..algorithm.agents import AgentGroup
from ..algorithm.model import RNNModel

class RolloutWorker():
    def __init__(self, env_config: EnvConfig, agent_group: AgentGroup, rnn_traj_len=5, device='cpu'):
        super(RolloutWorker, self).__init__()
        self.env_config = env_config
        self.agent_group = agent_group
        self.rnn_traj_len = rnn_traj_len
        self.env = self.env_config.create_env()
        self.device = device

    def generate_episode(self, episode_limit=None, epsilon=0.5):

        self.agent_group.eval().to(self.device)

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
                episode['states'].append(self.env.state())
                episode['actions'].append(actions)
                episode['avail_actions'].append(avail_actions)

                observations, rewards, terminations, truncations, infos = self.env.step(actions)
                
                episode['rewards'].append(rewards)
                episode['truncated'].append(truncations)
                episode['terminations'].append(terminations)
                episode['next_states'].append(self.env.state()) # TODO: Check if this is correct
                episode['next_observations'].append(observations) # TODO: Check if this is correct

                episode_reward += np.sum(np.array([rewards[agent] for agent in rewards.keys()]))

                # TODO win tag logic here
                # TODO logic for lost units
                # TODO reward of the last state and the second last state
                if True in terminations.values() or True in truncations.values():
                    break

            avail_actions = {agent: self.env.action_space(agent) for agent in self.env.agents}
            processed_obs = self._obs_preprocess(episode['observations']+[observations])
            actions = self.agent_group.act(processed_obs, avail_actions, epsilon)
            
        episode['episode_length'] = len(episode['observations'])
        episode['episode_reward'] = episode_reward
        episode['win_tag'] = win_tag

        # Close the environment after generating the episode
        self.env.close()

        return episode
    
    def _obs_preprocess(self, observations: list):
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
