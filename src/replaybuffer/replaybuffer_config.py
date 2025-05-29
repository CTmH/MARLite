from copy import deepcopy
from .normal_replaybuffer import NormalReplayBuffer
from .prioritized_replaybuffer import PrioritizedReplayBuffer
from .replaybuffer import ReplayBuffer

class ReplayBufferConfig:
    def __init__(self, **kwargs):
        self.config = deepcopy(kwargs)
        self.type = self.config.pop('type')
        self.registered_replaybuffers = {
            "Normal": NormalReplayBuffer,
            "Prioritized": PrioritizedReplayBuffer,
        }

    def create_replaybuffer(self) -> ReplayBuffer:
        if self.type not in self.registered_replaybuffers:
            raise ValueError(f"Unknown replaybuffer type: {self.type}")
        replaybuffer_class = self.registered_replaybuffers[self.type]
        return replaybuffer_class(**self.config)