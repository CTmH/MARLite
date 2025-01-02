from copy import deepcopy
from .normal_replaybuffer import NormalReplayBuffer

class ReplayBufferConfig:
    def __init__(self, **kwargs):
        self.config = deepcopy(kwargs)
        self.type = self.config.pop('type')
        self.registered_replaybuffers = {
            "Normal": NormalReplayBuffer,
        }

    def create_replaybuffer(self):
        replaybuffer_class = self.registered_replaybuffers[self.type]
        return replaybuffer_class(**self.config)