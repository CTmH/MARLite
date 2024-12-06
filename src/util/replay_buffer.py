import heapq
import random

from .trajectory_dataset import TrajectoryDataset

class ReplayBuffer:

    def __init__(self, capacity, traj_len):
        self.traj_len = traj_len
        self.capacity = capacity
        self.episode_buffer = []
        self.buffer = []

    def add_episode(self, episode):
        episode_id = len(self.episode_buffer)
        if episode['episode_length'] < 2:
            print(f"Episode {episode_id} is too short to be added to the replay buffer.")
            return self
        self.episode_buffer.append(episode)
        # TODO Need to remove old trajectories if buffer is full
        # Not included the last record because only the last state is used for training, last actions are not used.
        for i in range(episode['episode_length']-1):
            self.buffer.append((episode_id, i))
        return self
    '''
    def gen_traj(self, episode, episode_id):
        episode_length = len(episode)
        
        for i in range(episode_length):
            if i < self.traj_len - 1:
                # Not enough preceding elements to form a trajectory of length traj_len
                continue
            
            indices = tuple(range(i - (self.traj_len - 1), i + 1))
            self.buffer.append((episode_id, indices))

        return self
    '''
    def sample(self, sample_size):
        idx = random.sample(self.buffer, min(sample_size, len(self.buffer)))
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

