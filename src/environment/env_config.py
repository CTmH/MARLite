import pettingzoo as pz
from pettingzoo import ParallelEnv
from typing import Dict
import importlib

class EnvConfig():

    def __init__(self, *args, **kwargs) -> None:
        self.module_name = kwargs.get('module_name')
        self.env_name = kwargs.get('env_name')

        try:
            module = importlib.import_module(self.module_name)
            self.env = getattr(module, self.env_name)

        except (ImportError, AttributeError) as e:
            self.env = None
            print(f"Error loading environment {self.env_name} from module {self.module_name}: {e}")

    def create_env(self) -> ParallelEnv:
        return self.env.parallel_env()