from copy import deepcopy
from .rolloutmanager import RolloutManager
from .multiprocess_rolloutmanager import MultiProcessRolloutManager
from .multithread_rolloutmanager import MultiThreadRolloutManager


class RolloutManagerConfig:
    def __init__(self, **kwargs):
        self.config = deepcopy(kwargs)
        self.type = self.config.pop('type')
        self.registered_replaybuffers = {
            "multi-thread": MultiThreadRolloutManager,
            "multi-process": MultiProcessRolloutManager, # DO NOT USE, sitll need to debug
        }

    def create_replaybuffer(self):
        replaybuffer_class = self.registered_replaybuffers[self.type]
        return replaybuffer_class(**self.config)