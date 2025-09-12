from torch.utils.data import Dataset, DataLoader
import numpy as np

class TrajectoryDataset(Dataset):

    def __init__(self, sample_id_list, episode_buffer, traj_len):
        self.sample_id_list = sample_id_list
        self.episode_buffer = episode_buffer
        self.traj_len = traj_len
        self.attr = ['alive_mask',
                     'observations',
                     'states',
                     'edge_indices',
                     'next_states',
                     'next_observations',
                     'next_avail_actions',
                     'actions',
                     'rewards',
                     'terminations',
                     'truncations']

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
                if key == 'observations':
                    zero_obs = self.episode_buffer[episode_id][key][0]
                    zero_obs = {agent: np.zeros_like(o) for agent, o in zero_obs.items()}
                    sample[key].append(zero_obs)
                else:
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
        self.attr = ['alive_mask',
                     'observations',
                     'states',
                     'edge_indices',
                     'next_states',
                     'next_observations',
                     'next_avail_actions',
                     'actions',
                     'rewards',
                     'terminations',
                     'truncations']

    @staticmethod
    def collate_fn(batch):
        # Extract necessary components from the trajectory
        alive_mask = [traj['alive_mask'] for traj in batch]
        observations = [traj['observations'] for traj in batch]
        states = [traj['states'] for traj in batch]
        edge_indices = [traj['edge_indices'] for traj in batch]
        actions = [traj['actions'] for traj in batch]
        rewards = [traj['rewards'] for traj in batch]
        next_state = [traj['next_states'] for traj in batch]
        next_observations = [traj['next_observations'] for traj in batch]
        next_avail_actions = [traj['next_avail_actions'] for traj in batch]
        terminations = [traj['terminations'] for traj in batch]
        truncations = [traj['truncations'] for traj in batch]

        # Format Data

        # Observations
        # Nested list convert to numpy array (Batch Size, Time Step, Agent Number, Feature Dimensions) (B, T, N, F) -> (B, N, T, F)
        observations = np.array([[[value for _, value in dict.items()] for dict in traj] for traj in observations])
        next_observations = np.array([[[value for _, value in dict.items()] for dict in traj] for traj in next_observations])

        obs_shape = observations.shape
        n_dim_obs = len(obs_shape)
        transpose_arg = [0, 2, 1] + list(range(3, n_dim_obs))
        observations, next_observations = observations.transpose(transpose_arg), next_observations.transpose(transpose_arg)

        # Actions, Rewards, Terminations
        # Nested list convert to numpy array (Batch Size, Time Step, Agent Number) (B, T, N) -> (B, N, T)
        alive_mask = np.array([[[value for _, value in dict.items()] for dict in traj] for traj in alive_mask]).transpose(0,2,1)
        actions = np.array([[[value for _, value in dict.items()] for dict in traj] for traj in actions]).transpose(0,2,1)
        rewards = np.array([[[value for _, value in dict.items()] for dict in traj] for traj in rewards]).transpose(0,2,1)
        terminations = np.array([[[value for _, value in dict.items()] for dict in traj] for traj in terminations]).transpose(0,2,1)
        truncations = np.array([[[value for _, value in dict.items()] for dict in traj] for traj in truncations]).transpose(0,2,1)

        next_avail_actions =  np.array([[[value for _, value in dict.items()] for dict in traj] for traj in next_avail_actions])
        next_avail_actions = np.transpose(next_avail_actions, [0, 2, 1] if next_avail_actions.ndim == 3 else [0, 2, 1, 3])

        # States (Batch Size, Time Step, Feature Dimensions) (B, T, F)
        states = np.array(states)
        next_state = np.array(next_state)

        batch_dict = {
            'alive_mask': alive_mask,
            'observations': observations,
            'states': states,
            'edge_indices': edge_indices,
            'next_states': next_state,
            'next_observations': next_observations,
            'next_avail_actions': next_avail_actions,
            'actions': actions,
            'rewards': rewards,
            'terminations': terminations,
            'truncations': truncations
        }

        return batch_dict