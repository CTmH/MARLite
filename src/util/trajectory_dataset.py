from torch.utils.data import Dataset, DataLoader
import numpy as np

class TrajectoryDataset(Dataset):

    def __init__(self, sample_id_list, episode_buffer, traj_len):
        self.sample_id_list = sample_id_list
        self.episode_buffer = episode_buffer
        self.traj_len = traj_len
        self.attr = ['observations',
                     'states',
                     'next_states',
                     'next_observations',
                     'actions',
                     'rewards',
                     'terminations']

    def __len__(self):
        return len(self.sample_id_list)

    def __getitem__(self, idx):
        episode_id, pos = self.sample_id_list[idx]
        sample = {key: [] for key in self.attr}
        start = pos - self.traj_len + 1
        # Padding with the first element of the episode
        # if there is not enough elements in the episode before the start position
        while start < 0:
            for key in self.attr:
                sample[key].append(self.episode_buffer[episode_id][key][0])
            start += 1
        for key in self.attr:
            sample[key] += self.episode_buffer[episode_id][key][start:pos+1]

        return sample
    


class TrajectoryDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=0):
        super(TrajectoryDataLoader, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn
        )

    @staticmethod
    def collate_fn(batch):
        # Extract necessary components from the trajectory
        observations = [traj['observations'] for traj in batch]
        states = [traj['states'] for traj in batch]
        actions = [traj['actions'] for traj in batch]
        rewards = [traj['rewards'] for traj in batch]
        next_state = [traj['next_states'] for traj in batch]
        next_observations = [traj['next_observations'] for traj in batch]
        terminations = [traj['terminations'] for traj in batch]

        # Format Data

        # Observations
        # Nested list convert to numpy array (Batch Size, Time Step, Agent Number, Feature Dimensions) (B, T, N, F) -> (B, N, T, F)
        observations = [[[value for _, value in dict.items()] for dict in traj] for traj in observations]
        next_observations = [[[value for _, value in dict.items()] for dict in traj] for traj in next_observations]
        observations, next_observations = np.array(observations), np.array(next_observations)
        obs_shape = observations.shape
        n_dim_obs = len(obs_shape)
        transpose_arg = [0, 2, 1] + list(range(3, n_dim_obs))
        observations, next_observations = observations.transpose(transpose_arg), next_observations.transpose(transpose_arg)
        
        # Actions, Rewards, Terminations
        # Nested list convert to numpy array (Batch Size, Time Step, Agent Number) (B, T, N) -> (B, N, T)
        actions = [[[value for _, value in dict.items()] for dict in traj] for traj in actions]
        rewards = [[[value for _, value in dict.items()] for dict in traj] for traj in rewards]
        terminations = [[[value for _, value in dict.items()] for dict in traj] for traj in terminations]
        actions, rewards, terminations = np.array(actions), np.array(rewards), np.array(terminations)
        actions, rewards, terminations = actions.transpose(0,2,1), rewards.transpose(0,2,1), terminations.transpose(0,2,1)
        terminations = terminations.astype(int)  # Convert to int type for termination flags

        # States (Batch Size, Time Step, Feature Dimensions) (B, T, F)
        states = np.array(states)
        next_state = np.array(next_state)

        return observations, states, actions, rewards, next_state, next_observations, terminations