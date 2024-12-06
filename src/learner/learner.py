from typing import Dict

from module.agents import AgentGroup
from environment.env_config import EnvConfig
from module.model import ModelConfig
from rolloutworkers.worker import RolloutWorker

class Learner():
    def __init__(self, agents: Dict[str, str], env_config: EnvConfig, model_configs: ModelConfig):
        self.env_config = env_config
        self.model_configs = model_configs
        self.replay_buffer = None
        self.agents = agents
        self.target_agent_group = AgentGroup(agents=self.agents, env_config=self.env_config, model_configs=self.model_configs)
        # Set the same parameters for evaluation agents as target agents.
        self.target_models_params = self.target_agent_group.get_model_params()
    
    def learn(self):
        raise NotImplementedError
    
    def save_model(self):
        raise NotImplementedError
    
    def load_model(self):
        raise NotImplementedError
