import random
import numpy as np
from .replaybuffer import ReplayBuffer
from ..util.trajectory_dataset import TrajectoryDataset

class PrioritizedReplayBuffer(ReplayBuffer):

    def __init__(self, capacity, traj_len, priority_attr, alpha=0.8):
        super().__init__()
        self.priority_attr = priority_attr
        self.alpha = alpha
        self.traj_len = traj_len
        self.capacity = capacity
        self.episode_buffer = {i: None for i in range(self.capacity)}
        self.tail = -1

    def add_episode(self, episode):
        if episode['episode_length'] < 2:
            print("Episode is too short to be added to the replay buffer.")
            return self

        episode_id = self.tail + 1
        episode_id = episode_id % self.capacity
        self.tail = episode_id

        if self.episode_buffer[episode_id] is not None:
            self.remove_episode(episode_id)  # Remove old episode and old trajectory if it exists in the buffer.

        self.episode_buffer[episode_id] = episode
        # Add new trajectory position to the replay buffer.
        for i in range(episode['episode_length']):
            priority = episode[self.priority_attr][i]
            self.buffer.add((priority, episode_id, i))
        return self

    def remove_episode(self, episode_id):
        for i in range(self.episode_buffer[episode_id]['episode_length']):
            priority = self.episode_buffer[episode_id][self.priority_attr][i]
            self.buffer.remove((priority, episode_id, i))
        self.episode_buffer[episode_id] = None  # Remove the episode from the episode buffer
        return self

    def sample(self, sample_size):
        timestep_list = list(self.buffer)
        priorities = [ts[0] for ts in timestep_list]
        timestep_list = [ts[1:] for ts in timestep_list]
        min_priority = np.min(priorities)
        max_priority = np.max(priorities)
        if min_priority != max_priority:
            normalized_priorities = (priorities - min_priority) / (max_priority - min_priority) # Min-Max Normalization
        else:
            normalized_priorities = np.ones_like(priorities)
        weights = np.array(normalized_priorities) ** self.alpha  # 使用 alpha 调整优先级的影响
        # No need to manually normalize weights, random.choices will handle it.

        # Sample with replacement based on probabilities
        idx = random.choices(
            population=timestep_list,
            weights=weights,
            k=sample_size,
        )
        if len(idx) == 0:
            assert False, "Replay Buffer is empty"
        samples = TrajectoryDataset(sample_id_list=idx, episode_buffer=self.episode_buffer, traj_len=self.traj_len)
        return samples