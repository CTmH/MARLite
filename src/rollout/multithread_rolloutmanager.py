import threading
import queue
from typing import List, Any, Type
from ..algorithm.agents.agent_group import AgentGroup
from ..environment.env_config import EnvConfig
from .multiprocess_rolloutworker import MultiProcessRolloutWorker

class MultiThreadRolloutManager:
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

        self.worker_class = worker_class
        self.env_config = env_config
        self.agent_group = agent_group
        self.n_workers = n_workers
        self.n_episodes = n_episodes
        self.traj_len = traj_len
        self.episode_limit = episode_limit
        self.epsilon = epsilon
        self.device = device

        self.workers: List[threading.Thread] = []

        self.episode_queue = queue.Queue()

    def start(self):
        """启动所有工作线程"""
        episodes_per_worker = self.n_episodes // self.n_workers
        remaining_episodes = self.n_episodes % self.n_workers

        for i in range(self.n_workers):
            # 分配每个工作线程的任务量
            episode_count = episodes_per_worker
            if i < remaining_episodes:
                episode_count += 1

            # 创建工作线程
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
            thread = threading.Thread(target=worker.run)  # 假设 worker 有 run 方法
            self.workers.append(thread)
            thread.start()

    def join(self):
        """等待所有工作线程完成"""
        for worker in self.workers:
            worker.join()

    def generate_episodes(self) -> List[Any]:
        """生成并返回所有 episode 数据"""
        self.start()  # 启动工作线程
        self.join()   # 等待所有工作线程完成

        # 从队列中收集所有 episode 数据
        episodes = []
        while not self.episode_queue.empty():
            try:
                episodes.append(self.episode_queue.get_nowait())
            except queue.Empty:
                # 如果队列为空，退出循环
                break

        return episodes

    def cleanup(self):
        """清理资源"""
        for worker in self.workers:
            if worker.is_alive():
                # 线程没有 terminate 方法，只能通过标志位或其他方式停止
                pass
            worker.join()