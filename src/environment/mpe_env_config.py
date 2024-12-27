from typing import Dict
from pettingzoo import ParallelEnv
from pettingzoo.mpe import simple_spread_v3
from .env_config import EnvConfig

class MPEEnvConfig(EnvConfig):
    def __init__(self, env_config_dic: Dict) -> None:
        super().__init__(env_config_dic)

    def create_env(self) -> ParallelEnv:
        env = simple_spread_v3.parallel_env(render_mode="rgb_array")
        return env