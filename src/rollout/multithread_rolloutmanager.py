import threading
import queue
from typing import List, Any, Type, Optional
from ..algorithm.agents.agent_group import AgentGroup
from ..environment.env_config import EnvConfig
from .multithread_rolloutworker import MultiThreadRolloutWorker

class MultiThreadRolloutManager:
    def __init__(self,
                 worker_class: Type[MultiThreadRolloutWorker],
                 env_config: EnvConfig,
                 agent_group: AgentGroup,
                 n_workers: int,
                 n_episodes: int,
                 traj_len: int,
                 episode_limit: int,
                 epsilon: float,
                 device: str):

        # 参数校验
        if n_workers <= 0 or n_episodes <= 0:
            raise ValueError("n_workers and n_episodes must be positive integers")

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
        self.error_queue = queue.Queue()  # 异常收集队列

    def __enter__(self):
        """上下文管理器入口"""
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出时自动清理"""
        self.cleanup()

    def _distribute_episodes(self) -> List[int]:
        """分配每个worker的任务数量"""
        base = self.n_episodes // self.n_workers
        rem = self.n_episodes % self.n_workers
        return [base + (1 if i < rem else 0) for i in range(self.n_workers)]

    def _worker_wrapper(self, func) -> callable:
        """包装worker函数用于异常捕获"""
        def wrapped():
            try:
                func()
            except Exception as e:
                self.error_queue.put(e)
        return wrapped

    def start(self) -> "MultiThreadRolloutManager":
        """启动所有工作线程"""
        episodes_per_worker = self._distribute_episodes()

        for i, count in enumerate(episodes_per_worker):
            worker = self.worker_class(
                env_config=self.env_config,
                agent_group=self.agent_group.share_memory(),
                episode_queue=self.episode_queue,
                n_episodes=count,
                rnn_traj_len=self.traj_len,
                episode_limit=self.episode_limit,
                epsilon=self.epsilon,
                device=self.device
            )

            thread = threading.Thread(
                target=self._worker_wrapper(worker.run),
                name=f"RolloutWorker-{i}",
                daemon=True  # 守护线程确保主线程退出时自动终止
            )
            self.workers.append(thread)
            thread.start()
        return self

    def join(self) -> "MultiThreadRolloutManager":
        """等待所有工作线程完成"""
        for worker in self.workers:
            worker.join()
        
        # 检查是否有未处理的异常
        if not self.error_queue.empty():
            raise self.error_queue.get()
        return self

    def generate_episodes(self) -> List[Any]:
        """生成并返回所有episode数据"""
        self.start().join()

        # 可靠的数据收集方式
        episodes = []
        while True:
            try:
                episodes.append(self.episode_queue.get_nowait())
            except queue.Empty:
                break
        
        # 二次检查未处理异常
        if not self.error_queue.empty():
            raise self.error_queue.get()
        
        return episodes

    def cleanup(self) -> "MultiThreadRolloutManager":
        """清理资源"""
        # 如果worker支持停止机制
        if hasattr(self.worker_class, 'stop'):
            for worker in self.workers:
                if worker.is_alive():
                    worker.stop()
        
        # 等待所有线程结束
        for worker in self.workers:
            if worker.is_alive():
                worker.join(timeout=5)
        
        # 清空工作线程列表
        self.workers.clear()
        return self

    def get_progress(self) -> Optional[float]:
        """获取整体进度（可选实现）"""
        if not hasattr(self.worker_class, 'get_progress'):
            return None
        
        total = sum(
            worker.get_progress() 
            for worker in self.workers
            if worker.is_alive()
        ) / self.n_episodes
        return min(total, 1.0)