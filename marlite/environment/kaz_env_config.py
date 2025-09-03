from typing import Dict
from pettingzoo import ParallelEnv
from pettingzoo.butterfly import knights_archers_zombies_v10
from marlite.environment.env_config import EnvConfig

class KAZEnvConfig(EnvConfig):
    def __init__(self, env_config_dic: Dict) -> None:
        super().__init__(env_config_dic)

    def create_env(self) -> ParallelEnv:
        env = knights_archers_zombies_v10.parallel_env()
        return env