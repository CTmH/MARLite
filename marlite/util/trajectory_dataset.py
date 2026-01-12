import numpy as np
import torch
from operator import itemgetter
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate


ARRAY_ATTR = [
    'states',
    'edge_indices',
    'next_states',
]
DICT_ATTR = [
    'alive_mask',
    'observations',
    'next_observations',
    'next_avail_actions',
    'actions',
    'rewards',
    'terminations',
    'truncations',
]
PADDING_ATTR = [
    'obs_padding_mask',
    'next_obs_padding_mask',
]

NUMERIC_ATTR = [
    'states',
    'next_states',
    'alive_mask',
    'observations',
    'next_observations',
    'actions',
    'rewards',
    'terminations',
    'truncations',
    'obs_padding_mask',
    'next_obs_padding_mask',
]

DYNAMIC_LEN_ATTR = [
    'edge_indices',
]

OBJ_ATTR = [
    'next_avail_actions', # gym.spaces.Space if use_action_mask = False
]

class TrajectoryDataset(Dataset):

    def __init__(self, sample_id_list, episode_buffer, traj_len):
        self.sample_id_list = sample_id_list
        self.episode_buffer = episode_buffer
        self.traj_len = traj_len
        self.array_attr = ARRAY_ATTR
        self.dict_attr = DICT_ATTR
        self.padding_attr = PADDING_ATTR

    def __len__(self):
        return len(self.sample_id_list)

    def __getitem__(self, idx):
        episode_id, pos = self.sample_id_list[idx]
        sample = {key: [] for key in self.array_attr + self.dict_attr + self.padding_attr}
        start = pos - self.traj_len + 1

        # Padding with the first element of the episode
        # if there is not enough elements in the episode before the start position
        while start < 0:
            # Handle array attributes
            for key in self.array_attr:
                sample[key].append(np.zeros_like(self.episode_buffer[episode_id][key][0]))

            # Handle dictionary attributes
            for key in self.dict_attr:
                first_element = self.episode_buffer[episode_id][key][0]
                zero_element = np.stack([np.zeros_like(value) for value in first_element.values()])
                sample[key].append(zero_element)

            for key in self.padding_attr:
                sample[key].append(True)
            start += 1

        # Process array attributes (no conversion needed)
        for key in self.array_attr:
            sample[key] += self.episode_buffer[episode_id][key][start:pos+1]

        # Process dictionary attributes (convert to numpy arrays)
        for key in self.dict_attr:
            dict_sequence = self.episode_buffer[episode_id][key][start:pos+1]
            # Convert each dictionary in the sequence to a numpy array
            converted_sequence = []
            for dict_item in dict_sequence:
                # Extract values in order and stack them into a single array
                converted_array = np.stack(list(dict_item.values()), axis=0)  # Stack along agent dimension
                converted_sequence.append(converted_array)
            sample[key] += converted_sequence

        for key in self.padding_attr:
            sample[key] += [False] * (pos - start + 1)

        return sample

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class TrajectoryDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=0):
        super(TrajectoryDataLoader, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=trajectory_collate_fn
        )

def trajectory_collate_fn(batch):
    """
    Custom collate for TrajectoryDataset that preserves gym.Space objects.

    Assumes each sample is a dict like:
        {
            'actions': np.ndarray,
            'observations': np.ndarray,
            ...
            'action_space': Discrete(5),   # ← Space stored under reserved key
        }

    Args:
        batch: List of samples from Dataset.__getitem__ (each is a dict)

    Returns:
        collated: dict where numeric arrays are stacked by default_collate,
                  and Space objects are kept as-is (one per batch, assumed identical).
    """
    # Collate numeric data using PyTorch's default behavior
    #numeric_batch = [dict(zip(NUMERIC_ATTR, itemgetter(*NUMERIC_ATTR)(sample))) for sample in batch]
    #collated = default_collate(numeric_batch)
    collated = {}
    for k in NUMERIC_ATTR:
        collated[k] = torch.tensor([sample[k] for sample in batch])

    # Preserve space objects — assume they're identical across batch
    for k in OBJ_ATTR:
        first_elem = batch[0][k][0]
        if np.issubdtype(first_elem.dtype, np.object_):
            collated[k] = np.stack([sample[k] for sample in batch])  # Keep original object (not collated)
        elif np.issubdtype(first_elem.dtype, np.number):
            collated[k] = torch.tensor([sample[k] for sample in batch])
        else:
            raise ValueError(f"Unexpected data type for {k}: {batch[0][k][0].dtype}")

    for k in DYNAMIC_LEN_ATTR:
        collated[k] = [sample[k] for sample in batch]

    return collated