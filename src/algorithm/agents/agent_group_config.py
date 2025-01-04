import torch
from copy import deepcopy
from .qmix_agent_group import QMIXAgentGroup
from .random_agent_group import RandomAgentGroup
from ..model import ModelConfig
from ...util.optimizer_config import OptimizerConfig

def get_qmix_agent_group(agent_group_config):
    agents = agent_group_config["agent_list"]
    text_model_configs = agent_group_config["model_configs"]
    model_configs = {}
    feature_extractor_configs = {}
    for model, conf in text_model_configs.items():
        model_conf = deepcopy(conf)
        if 'feature_extractor' in conf:  # Check if feature extractor is defined in the model configuration. If not, use Identity as default.
            fe_conf = model_conf.pop('feature_extractor')
        else:
            fe_conf = {'model_type': 'Identity'}
        model_configs[model] = ModelConfig(**model_conf)
        feature_extractor_configs[model] = ModelConfig(**fe_conf)
    optimizer_config = agent_group_config["optimizer"]
    optimizer_config = OptimizerConfig(**optimizer_config)
    return QMIXAgentGroup(agents, model_configs, feature_extractor_configs, optimizer_config)

def get_random_agent_group(agent_group_config):
    agents = agent_group_config["agent_list"]
    return RandomAgentGroup(agents)

registered_agent_groups = {
    "QMIX": get_qmix_agent_group,
    "Random": get_random_agent_group
}

class AgentGroupConfig(object):
    def __init__(self, **kwargs):
        self.agent_group_config = deepcopy(kwargs)
        self.ag_type = self.agent_group_config.pop("type")
        if self.ag_type not in registered_agent_groups:
            raise ValueError(f"Agent group type {self.ag_type} not registered.")
        
    def get_agent_group(self):
        return registered_agent_groups[self.ag_type](self.agent_group_config)