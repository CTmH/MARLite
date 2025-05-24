import numpy as np
import threading
import queue
import logging
from copy import deepcopy
from ..environment.env_config import EnvConfig
from ..algorithm.agents import AgentGroup
from ..algorithm.agents.gnn_agent_group import GNNAgentGroup
from ..algorithm.model import TimeSeqModel

class MultiThreadRolloutWorker(threading.Thread):
    def __init__(self,
                 env_config: EnvConfig,
                 agent_group: AgentGroup,
                 episode_queue: queue.Queue,
                 n_episodes: int,
                 rnn_traj_len=5,
                 episode_limit=100,
                 epsilon=0.5,
                 device='cpu'):
        super(MultiThreadRolloutWorker, self).__init__()
        self.env_config = env_config
        self.agent_group = deepcopy(agent_group)
        self.episode_queue = episode_queue
        self.n_episodes = n_episodes
        self.rnn_traj_len = rnn_traj_len
        self.episode_limit = episode_limit
        self.epsilon = epsilon
        self.env = self.env_config.create_env()
        self.device = device

        self.thread_name = threading.current_thread().name
        self.thread_id = threading.current_thread().ident

    def run(self):
        for i in range(self.n_episodes):
            self.episode_queue.put(self.rollout())
            if self.n_episodes < 10 or i % (self.n_episodes // 10) == 0 or i == (self.n_episodes - 1):
                logging.info(f"Thread - {self.thread_id}:\t{self.thread_name}\tfinished job {i+1} / {self.n_episodes}")
        return self

    def rollout(self):
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

                episode_reward += np.sum(np.array([reward for _, reward in rewards.items()]))

                # TODO win tag logic here
                # TODO logic for lost units
                # TODO reward of the last state and the second last state
                if True in terminations.values() or True in truncations.values():
                    break

            avail_actions = {agent: self.env.action_space(agent) for agent in self.env.agents}
            processed_obs = self._obs_preprocess(episode['observations']+[observations])
            if isinstance(self.agent_group, GNNAgentGroup):
                _, edge_index = self.env.build_my_team_graph()
                actions = self.agent_group.act(processed_obs, edge_index, avail_actions, self.epsilon)
            else:
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
                obs_len = len(observations[-self.rnn_traj_len:])
                if obs_len < self.rnn_traj_len:
                    padding_length = self.rnn_traj_len - obs_len
                    obs_padding = [np.zeros_like(observations[-1][agent_id]) for _ in range(padding_length)]
                    obs = obs_padding + [o[agent_id] for o in observations[-self.rnn_traj_len:]]
                else:
                    obs = [o[agent_id] for o in observations[-self.rnn_traj_len:]]
            else:
                obs = observations[-1][agent_id]
            processed_obs[agent_id] = np.array(obs)
        return processed_obs