import pettingzoo as pz
from pettingzoo import ParallelEnv
from typing import Dict

class EnvConfig():
    def __init__(self, env_config_dic: Dict) -> None:
        self.env_config_dic = env_config_dic

    def create_env(self) -> ParallelEnv:
        raise NotImplementedError