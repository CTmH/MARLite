import numpy as np
from typing import List, Union
from numpy import ndarray
from marlite.algorithm.group_builder.group_builder import GroupBuilder


class FixedGroupBuilder(GroupBuilder):
    def __init__(self, group_ids: List[int], dtype=np.int16):
        super().__init__(dtype=dtype)
        self.group_ids = np.array(group_ids, dtype=dtype)

    def forward(self, states: ndarray) -> ndarray:
        bs = states.shape[0]
        zone_indices = np.tile(self.group_ids[np.newaxis, :], (bs, 1))
        return zone_indices.astype(self.dtype)

    def reset(self):
        return self
