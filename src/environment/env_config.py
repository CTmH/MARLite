import pettingzoo as pz
from pettingzoo import ParallelEnv
from typing import Dict
import importlib

from .parallel_env_wrapper import ParallelEnvWrapper
from .adversarial_pursuit_wrapper import AdversarialPursuitPredator

custom_envs = {
    'adversarial_pursuit_predator': AdversarialPursuitPredator
}

class EnvConfig():

    def __init__(self, **kwargs) -> None:
        self.module_name = kwargs.pop('module_name')
        self.env_name = kwargs.pop('env_name')
        self.env_config = kwargs

        if self.module_name != 'custom':
            try:
                module = importlib.import_module(self.module_name)
                self.env = getattr(module, self.env_name)
            except (ImportError, AttributeError) as e:
                self.env = None
                print(f"Error loading environment {self.env_name} from module {self.module_name}: {e}")
        elif self.env_name in custom_envs:
            self.env = custom_envs[self.env_name]
        else:
            raise ValueError(f"Custom environment {self.env_name} not registered.")

    def create_env(self) -> ParallelEnv:
        if self.module_name != 'custom':
            return self.env.parallel_env()
        else:
            return self.env(**self.env_config)