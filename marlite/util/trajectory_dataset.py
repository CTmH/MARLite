import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Used by TrajectoryDataset.__getitem__ to build sample dict keys
FLOAT_ATTR = [
    "states",
    "next_states",
    "observations",
    "next_observations",
]

# Used by TrajectoryDataset.__getitem__ for array-type episode buffer fields
ARRAY_ATTR = [
    "states",
    "next_states",
    "edge_indices",
    "next_edge_indices",
]

# Used by TrajectoryDataset.__getitem__ for dict-type episode buffer fields
DICT_ATTR = [
    "alive_mask",
    "observations",
    "next_observations",
    "next_alive_mask",
    "next_avail_actions",
    "actions",
    "all_log_probs",
    "log_probs",
    "rewards",
    "terminations",
    "truncations",
    "group_indices",
    "next_group_indices",
]

# Used by TrajectoryDataset.__getitem__ for padding mask fields
PADDING_ATTR = [
    "timestep_padding_mask",
    "next_timestep_padding_mask",
]

# Used by trajectory_collate_fn to stack samples into tensors
NUMERIC_ATTR = [
    "states",
    "next_states",
    "alive_mask",
    "observations",
    "actions",
    "all_log_probs",
    "log_probs",
    "rewards",
    "terminations",
    "truncations",
    "timestep_padding_mask",
    "next_alive_mask",
    "next_observations",
    "next_timestep_padding_mask",
    "group_indices",
    "next_group_indices",
    "formatted_obs",
    "construct_padding_mask",
]

# Used by trajectory_collate_fn for variable-length fields (kept as list)
DYNAMIC_LEN_ATTR = [
    "edge_indices",
    "next_edge_indices",
]

# Used by trajectory_collate_fn to preserve gym.Space objects
OBJ_ATTR = [
    "next_avail_actions",
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
        sample = {
            key: [] for key in self.array_attr + self.dict_attr + self.padding_attr
        }
        start = pos - self.traj_len + 1

        # Padding with the first element of the episode
        # if there is not enough elements in the episode before the start position
        while start < 0:
            # Handle array attributes
            for key in self.array_attr:
                sample[key].append(
                    np.zeros_like(self.episode_buffer[episode_id][key][0])
                )

            # Handle dictionary attributes
            for key in self.dict_attr:
                first_element = self.episode_buffer[episode_id][key][0]
                zero_element = np.stack(
                    [np.zeros_like(value) for value in first_element.values()]
                )
                sample[key].append(zero_element)

            for key in self.padding_attr:
                sample[key].append(True)
            start += 1

        # Process array attributes (no conversion needed)
        for key in self.array_attr:
            sample[key] += self.episode_buffer[episode_id][key][start : pos + 1]

        # Process dictionary attributes (convert to numpy arrays)
        for key in self.dict_attr:
            dict_sequence = self.episode_buffer[episode_id][key][start : pos + 1]
            # Convert each dictionary in the sequence to a numpy array
            converted_sequence = []
            for dict_item in dict_sequence:
                # Extract values in order and stack them into a single array
                converted_array = np.stack(
                    list(dict_item.values()), axis=0
                )  # Stack along agent dimension
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
            collate_fn=trajectory_collate_fn,
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
    collated = {}
    first_sample_keys = set(batch[0].keys())

    numeric_keys = [k for k in NUMERIC_ATTR if k in first_sample_keys]
    obj_keys = [k for k in OBJ_ATTR if k in first_sample_keys]
    dynamic_keys = [k for k in DYNAMIC_LEN_ATTR if k in first_sample_keys]

    for k in numeric_keys:
        batch_values = [sample[k] for sample in batch]
        collated[k] = torch.tensor(np.array(batch_values))

    for k in obj_keys:
        first_elem = batch[0][k][0]
        if np.issubdtype(first_elem.dtype, np.object_):
            collated[k] = np.stack([sample[k] for sample in batch])
        elif np.issubdtype(first_elem.dtype, np.number):
            collated[k] = torch.tensor([sample[k] for sample in batch])
        else:
            raise ValueError(f"Unexpected data type for {k}: {first_elem.dtype}")

    for k in dynamic_keys:
        collated[k] = [sample[k] for sample in batch]

    return collated


class SSLEnrichedTrajectoryDataset(Dataset):
    """
    A TrajectoryDataset wrapper that precomputes and enriches samples with SSL data.

    Instead of computing SSL data per-batch in collate_fn (which is slow),
    this dataset:
    1. Materializes all samples via list(trajectory_dataset)
    2. Extracts SSL-relevant fields (observations, states, edge_indices, alive_mask)
    3. Processes all SSL data in a single call to data_constructor.process()
    4. Enriches each sample dict with formatted_obs and construct_padding_mask
    5. Returns enriched samples that work with standard TrajectoryDataLoader

    This approach is faster because:
    - SSL data construction happens once, not per-batch
    - data_constructor.process() can process all samples together (batch processing)
    - Collate_fn just converts to tensors without complex processing
    """

    def __init__(self, trajectory_dataset, data_constructor):
        """
        Initialize SSLEnrichedTrajectoryDataset.

        Args:
            trajectory_dataset: TrajectoryDataset instance to wrap
            data_constructor: SelfSupervisedDataConstructor instance for SSL preprocessing
        """
        self.base_dataset = trajectory_dataset
        self.data_constructor = data_constructor
        self.traj_len = trajectory_dataset.traj_len

        self._enrich_all_samples()

    def _enrich_all_samples(self):
        """
        Materialize all samples and compute SSL data for all at once.
        """
        all_samples = list(self.base_dataset)

        observations = np.array([s["observations"] for s in all_samples])
        states = np.array([s["states"] for s in all_samples])
        alive_masks = np.array([s["alive_mask"] for s in all_samples])
        edge_indices = [s["edge_indices"] for s in all_samples]

        formatted_obs, construct_padding_mask = self.data_constructor.process(
            observations, states, edge_indices, alive_masks
        )

        for i, sample in enumerate(all_samples):
            sample["formatted_obs"] = formatted_obs[i]
            sample["construct_padding_mask"] = construct_padding_mask[i]

        self._enriched_samples = all_samples

    def __len__(self):
        return len(self._enriched_samples)

    def __getitem__(self, idx):
        return self._enriched_samples[idx]
