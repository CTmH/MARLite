from typing import Dict, Any
from pettingzoo import ParallelEnv
import importlib
from copy import deepcopy

from marlite.environment.magent_wrapper import AdversarialPursuitPredator, AdversarialPursuitPrey, BattleWrapper
from marlite.environment.smac_wrapper import SMACWrapper
from marlite.environment.sumo_wrapper import SUMOWrapper

REGISTERED_WRAPPERS = {
    'adversarial_pursuit_predator': AdversarialPursuitPredator,
    'adversarial_pursuit_prey': AdversarialPursuitPrey,
    'battle': BattleWrapper,
    'battlefield': BattleWrapper,
    'smac': SMACWrapper,
    'sumo': SUMOWrapper
}

class EnvConfig():

    def __init__(self, module_name: str, env_name: str, env_config: Dict[str, Any] | None = None,
                 wrapper: Dict[str, Any] | None = None, wrapper_config: Dict[str, Any] | None = None) -> None:
        self.module_name = module_name
        self.env_name = env_name
        self.env_config = env_config or {}
        wrapper_dict = wrapper if wrapper is not None else wrapper_config
        self.wrapper = deepcopy(wrapper_dict) if wrapper_dict else None
        self.wrapper_type = None
        if self.wrapper:
            self.wrapper_type = self.wrapper.pop('type', None)
            if self.wrapper_type not in REGISTERED_WRAPPERS:
                raise ValueError(f"Unknown wrapper type: {self.wrapper_type}")

    def create_env(self) -> ParallelEnv:
        try:
            importlib.import_module(f'{self.module_name}.{self.env_name}')
            module = importlib.import_module(self.module_name)
            env_class = getattr(module, self.env_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Error loading environment {self.env_name} from module {self.module_name}: {e}")

        if self.env_config:
            env:ParallelEnv = env_class.parallel_env(**self.env_config)
        else:
            env:ParallelEnv = env_class.parallel_env()
        if self.wrapper_type:
            wrapper_params = self.wrapper.copy()
            wrapper_params["opponent_agent_group_config"] = wrapper_params.pop(
                "opponent_agent_group", None
            )
            wrapper_class = REGISTERED_WRAPPERS[self.wrapper_type]
            env = wrapper_class(env, **wrapper_params)

        return env
