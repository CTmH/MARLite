import multiprocessing
import os
import logging
import numpy as np
import torch.multiprocessing as mp
from copy import deepcopy
from ..environment.env_config import EnvConfig
from ..algorithm.agents import AgentGroup
from ..algorithm.model import TimeSeqModel
from ..algorithm.agents.agent_group import AgentGroup

class MultiProcessRolloutWorker(mp.Process):
    def __init__(self,
                 env,
                 agent_group: AgentGroup,
                 n_episodes: int,
                 rnn_traj_len=5,
                 episode_limit=100,
                 epsilon=0.5,
                 device='cpu'):
        super(MultiProcessRolloutWorker, self).__init__()
        self.device = device
        self.env = env
        self.episode_list = []
        # Initialize AgentGroup
        self.agent_group = agent_group
        self.agent_group.to(self.device)
        self.n_episodes = n_episodes
        self.rnn_traj_len = rnn_traj_len
        self.episode_limit = episode_limit
        self.epsilon = epsilon

        self.process_name = multiprocessing.current_process().name
        self.process_id = os.getpid()

    def run(self):
        for i in range(self.n_episodes):
            self.episode_list.append(self.rollout())
            if self.n_episodes < 10 or i % (self.n_episodes // 10) == 0 or i == (self.n_episodes - 1):
                logging.info(f"Process - {self.process_id}:\t{self.process_name}\tfinished job {i+1} / {self.n_episodes}")
        return deepcopy(self.episode_list)

    def rollout(self):

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
                observations, infos = self.env.reset()
            else:
                episode['observations'].append(observations)
                episode['states'].append(self.env.state())
                episode['actions'].append(actions)
                episode['avail_actions'].append(avail_actions)

                observations, rewards, terminations, truncations, infos = self.env.step(actions)
                
                episode['rewards'].append(rewards)
                all_agents_rewards = [value for _, value in rewards.items()]
                episode['all_agents_sum_rewards'].append(sum(all_agents_rewards))
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
            actions = self.agent_group.act(processed_obs, avail_actions, self.epsilon)
            
        episode['episode_length'] = len(episode['observations'])
        episode['episode_reward'] = episode_reward
        episode['win_tag'] = win_tag

        # Close the environment after generating the episode
        self.env.close()

        return episode
    
    def _obs_preprocess(self, observations: list):
        agents = self.env.agents
        models = self.agent_group.models
        agent_model_dict = self.agent_group.agent_model_dict
        processed_obs = {agent_id : [] for agent_id in agents}
        for agent_id, model_name in agent_model_dict.items():
            if isinstance(models[model_name], TimeSeqModel):
                obs = [o[agent_id] for o in observations[-self.rnn_traj_len:]]
            else:
                obs = observations[-1][agent_id]
            processed_obs[agent_id] = np.array(obs)
        return processed_obs