import heapq
import random

from .trajectory_dataset import TrajectoryDataset

class ReplayBuffer:

    def __init__(self, capacity, traj_len):
        self.traj_len = traj_len
        self.capacity = capacity
        self.episode_buffer = {i: None for i in range(self.capacity)}
        self.tail = -1
        self.buffer = set()

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
            self.buffer.add((episode_id, i))
        return self
    
    def remove_episode(self, episode_id):
        for i in range(self.episode_buffer[episode_id]['episode_length']):
            self.buffer.remove((episode_id, i))
        self.episode_buffer[episode_id] = None  # Remove the episode from the episode buffer
        return self

    def sample(self, sample_size):
        idx = random.sample(list(self.buffer), min(sample_size, len(self.buffer)))
        if len(idx) == 0:
            assert False, "Replay Buffer is empty"
        samples = TrajectoryDataset(sample_id_list=idx, episode_buffer=self.episode_buffer, traj_len=self.traj_len)
        return samples


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
    
    def add(self, priority, experience):
        if len(self.buffer) >= self.capacity:
            # Remove the lowest priority item if buffer is full
            heapq.heappop(self.buffer)
        heapq.heappush(self.buffer, (priority, experience))
    
    def sample(self, batch_size):
        sampled_experiences = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return [exp for _, exp in sorted(sampled_experiences, key=lambda x: x[0])]
    
    def update_priority(self, index, new_priority):
        # Find the experience with the old priority
        for i, (priority, exp) in enumerate(self.buffer):
            if i == index:
                # Remove the old one
                self.buffer[i] = None
                heapq.heapify(self.buffer)
                # Add the new one
                self.add(new_priority, exp)
                break
    
    def __len__(self):
        return len([item for item in self.buffer if item is not None])

