from pettingzoo import ParallelEnv
import importlib

from .adversarial_pursuit_wrapper import AdversarialPursuitPredator, AdversarialPursuitPrey
from .battle_wrapper import BattleWrapper
from .battlefield_wrapper import BattleFieldWrapper

CUSTOM_ENVS = {
            'adversarial_pursuit_predator': AdversarialPursuitPredator,
            'adversarial_pursuit_prey': AdversarialPursuitPrey,
            'battle': BattleWrapper,
            'battlefield': BattleFieldWrapper,
        }

class EnvConfig():

    def __init__(self, **kwargs) -> None:
        self.module_name = kwargs.pop('module_name')
        self.env_name = kwargs.pop('env_name')
        self.env_config = kwargs

    def create_env(self) -> ParallelEnv:

        if self.module_name != 'custom':
            try:
                importlib.import_module(f'{self.module_name}.{self.env_name}')
                module = importlib.import_module(self.module_name)
                env = getattr(module, self.env_name)
            except (ImportError, AttributeError) as e:
                env = None
                raise ValueError(f"Error loading environment {self.env_name} from module {self.module_name}: {e}")
        elif self.env_name in CUSTOM_ENVS:
            env = CUSTOM_ENVS[self.env_name]
        else:
            raise ValueError(f"Custom environment {self.env_name} not registered.")

        if self.module_name != 'custom':
            return env.parallel_env(**self.env_config)
        else:
            return env(**self.env_config)