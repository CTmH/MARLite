import multiprocessing as mp
from typing import List, Any, Type
from copy import deepcopy
from ..algorithm.agents.agent_group import AgentGroup
from ..environment.env_config import EnvConfig
from .rolloutmanager import RolloutManager

def rolloutworker_process(pipe,
                 env_config: EnvConfig,
                 ):
    env_config = env_config
    env = env_config.create_env()
    while True:
        # 等待主进程发送动作
        action = pipe.recv()
        if action is None:  # 接收到None表示结束
            break
        elif action == 'reset':
            observations, infos = env.reset()
            state = env.state()
            pipe.send((state, observations, infos))
        else:
            # 执行动作并返回奖励
            observations, rewards, terminations, truncations, infos = env.step(action)
            state = env.state()
            pipe.send((state, observations, rewards, terminations, truncations, infos))

class MultiProcessRolloutManager(RolloutManager):
    def __init__(self,
                 env_config: EnvConfig,
                 agent_group: AgentGroup,
                 n_workers: int,
                 n_episodes: int,
                 traj_len: int,
                 episode_limit: int,
                 epsilon: float,
                 device: str):

        self.env_config = env_config
        self.agent_group = agent_group
        self.n_workers = n_workers
        self.n_episodes = n_episodes
        self.traj_len = traj_len
        self.episode_limit = episode_limit
        self.epsilon = epsilon
        self.device = device
        self.env = self.env_config.create_env()

        self.workers = []
        self.pipes = []

        for _ in range(self.n_workers):
            parent_conn, child_conn = mp.Pipe()
            p = mp.Process(target=rolloutworker_process, args=(child_conn,))
            p.start()
            self.workers.append(p)
            self.pipes.append(parent_conn)

    def start(self):

        episodes_per_worker = self.n_episodes // self.n_workers
        remaining_episodes = self.n_episodes % self.n_workers

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
        batch = {key: [] for key in episode.keys()}

        for i in range(episodes_per_worker):
            episode_batch = [deepcopy(episode) for _ in range(self.n_workers)]

            # Reset Env
            observation_batch = []
            state_batch = []
            for pipe in self.pipes:
                pipe.send('reset')
            for pipe in self.pipes:
                state, observation, infos = pipe.recv()
                observation_batch.append(observation)
                state_batch.append(state)

            for j in range(self.episode_limit):
                avail_actions = {agent: self.env.action_space(agent) for agent in self.env.agents}
                batch_avail_actions = [avail_actions for _ in range(self.n_workers)]
                window = batch['observations'] + [observation_batch]
                window = window[-self.traj_len:]
                # window shape: (T, B, N, (obs_shape))
                # Preprocess observations for each worker and agent across timesteps.
                processed_obs = [
                    {
                        agent_id: [window[t][i][agent_id] for t in range(len(window))]
                        for agent_id in self.env.agents
                    }
                    for i in range(self.n_workers)
                ]
                # processed_obs shape: (B, N, T, (obs_shape))
                batch_actions = self.agent_group.batch_act(processed_obs, batch_avail_actions)
                batch['observations'].append(observation_batch)
                batch['states'].append(state_batch)
                batch['actions'].append(batch_actions)
                batch['avail_actions'].append(batch_avail_actions)

                observation_batch = []
                rewards_batch = []
                terminations_batch = []
                truncations_batch = []
                for worker in self.workers:
                    observations, rewards, terminations, truncations, infos = worker.recv()
                    observation_batch.append(observations)
                    rewards_batch.append(rewards)
                    terminations_batch.append(terminations)
                    truncations_batch.append(truncations)

                
            
        return self

    def join(self):
        """等待所有工作进程完成"""
        for worker in self.workers:
            worker.join()
        return self

    def generate_episodes(self) -> List[Any]:
        """生成并返回所有 episode 数据"""
        self.start()  # 启动工作进程
        self.join()   # 等待所有工作进程完成

        # 从队列中收集所有 episode 数据
        episodes = []
        while not self.episode_queue.empty():
            try:
                episodes.append(self.episode_queue.get_nowait())
            except mp.queues.Empty:
                # 如果队列为空，退出循环
                break

        return episodes

    def cleanup(self):
        """清理资源"""
        for worker in self.workers:
            if worker.is_alive():
                worker.terminate()
            worker.join()
        return self
