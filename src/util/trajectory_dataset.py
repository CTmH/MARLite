from torch.utils.data import Dataset, DataLoader

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
    
    def collate_fn(batch):
        return batch


    @staticmethod
    def collate_fn(batch):
        # Assuming each item in the batch is a dictionary with lists as values
        result = {}
        for key in batch[0].keys():
            result[key] = [item[key] for item in batch]
        return result

# TODO Costomize DataloaderIter 
'''
# 自定义DataLoader的迭代器
class TrajectoryDataLoaderIter:
    def __init__(self, loader):
        self.loader = loader
        self.dataset = loader.dataset
        self.batch_size = loader.batch_size
        self.num_workers = loader.num_workers
        self.sampler = loader.sampler
        self.batch_sampler = loader.batch_sampler
        self.sample_iter = iter(self.sampler)
        self.reset()

    def reset(self):
        self.sample_iter = iter(self.sampler)

    def __next__(self):
        if self.num_workers > 0:
            r = self._process_data()
        else:
            r = self._process_data_serial()
        if r is None:
            raise StopIteration
        return r

    def _process_data(self):
        # 自定义的数据处理逻辑
        try:
            indices = next(self.sample_iter)
        except StopIteration:
            self.reset()
            raise StopIteration

        batch = []
        for idx in indices:
            sample = self.dataset[idx]
            batch.append(sample)

        return batch

    def _process_data_serial(self):
        try:
            idx = next(self.sample_iter)
        except StopIteration:
            self.reset()
            raise StopIteration

        sample = self.dataset[idx]
        return sample
'''