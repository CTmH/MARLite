import torch.multiprocessing as mp
from multiprocessing import queues
from typing import List, Any, Type
from ..algorithm.agents.agent_group import AgentGroup
from ..environment.env_config import EnvConfig
from .multiprocess_rolloutworker import MultiProcessRolloutWorker

class MultiProcessRolloutManager:
    def __init__(self,
                 worker_class: Type[MultiProcessRolloutWorker],
                 env_config: EnvConfig,
                 agent_group: AgentGroup,
                 n_workers: int,
                 n_episodes: int,
                 traj_len: int,
                 episode_limit: int,
                 epsilon: float,
                 device: str):

        # 使用spawn启动方法确保CUDA兼容性
        mp.set_start_method('spawn', force=True)
        
        self.worker_class = worker_class
        self.env_config = env_config
        self.agent_group = agent_group
        self.n_workers = n_workers
        self.n_episodes = n_episodes
        self.traj_len = traj_len
        self.episode_limit = episode_limit
        self.epsilon = epsilon
        self.device = device

        self.workers: List[mp.Process] = []
        self.episode_queue = mp.Queue()  # 多进程安全队列

    def start(self):
        """启动所有工作进程"""
        episodes_per_worker = self.n_episodes // self.n_workers
        remaining_episodes = self.n_episodes % self.n_workers

        for i in range(self.n_workers):
            episode_count = episodes_per_worker
            if i < remaining_episodes:
                episode_count += 1

            # 创建并启动工作进程
            worker = self.worker_class(
                env_config=self.env_config,
                agent_group=self.agent_group,
                episode_queue=self.episode_queue,
                n_episodes=episode_count,
                rnn_traj_len=self.traj_len,
                episode_limit=self.episode_limit,
                epsilon=self.epsilon,
                device=self.device
            )
            process = mp.Process(target=worker.run)
            self.workers.append(process)
            process.start()
        return self

    def join(self):
        """等待所有工作进程完成"""
        for worker in self.workers:
            worker.join()
        return self

    def generate_episodes(self) -> List[Any]:
        """生成并返回所有episode数据"""
        self.start()
        self.join()

        # 收集所有episode数据
        episodes = []
        while True:
            try:
                # 设置超时防止死锁
                episode = self.episode_queue.get(block=False, timeout=0.1)
                episodes.append(episode)
            except (queues.Empty, KeyboardInterrupt):
                break
        return episodes

    def cleanup(self):
        """确保终止所有子进程"""
        for worker in self.workers:
            if worker.is_alive():
                worker.terminate()
            worker.close()  # 添加进程资源释放
        self.episode_queue.cancel_join_thread()  # 防止队列线程阻塞
        self.episode_queue.close()
        return self