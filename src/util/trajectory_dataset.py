from torch.utils.data import Dataset, DataLoader

class TrajectoryDataset(Dataset):

    def __init__(self, sample_id_list, episode_buffer, traj_len):
        self.sample_id_list = sample_id_list
        self.episode_buffer = episode_buffer
        self.traj_len = traj_len
        self.attr = ['observations', 'state', 'actions', 'rewards']

    def __len__(self):
        return len(self.sample_id_list)

    def __getitem__(self, idx):
        episode_id, pos = self.sample_id_list[idx]
        sample = {key: [] for key in self.attr}
        start = pos - self.traj_len + 1
        # Padding with the first element of the episode if there is not enough elements in the episode before the start position
        while start < 0:
            for key in self.attr:
                sample[key].append(self.episode_buffer[episode_id][key][0])
            start += 1
        for key in self.attr:
            sample[key] += self.episode_buffer[episode_id][key][start:pos+1]

        return sample

class TrajectoryDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.dataset = dataset