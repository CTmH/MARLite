from copy import deepcopy
from typing import Dict, Any
from marlite.algorithm.agents.agent_group import AgentGroup
from marlite.algorithm.agents.qmix_agent_group import QMIXAgentGroup
from marlite.algorithm.agents.gnn_agent_group import GNNAgentGroup
from marlite.algorithm.agents.random_agent_group import RandomAgentGroup
from marlite.algorithm.agents.magent_agent_group import MagentPreyAgentGroup, MagentBattleAgentGroup
from marlite.algorithm.agents.msg_aggr_agent_group import MsgAggrAgentGroup
from marlite.algorithm.agents.gnn_obs_comm_agent_group import GNNObsCommAgentGroup
from marlite.algorithm.model import ModelConfig
from marlite.algorithm.graph_builder import GraphBuilderConfig
from marlite.util.optimizer_config import OptimizerConfig

def get_qmix_agent_group(agent_group_config: Dict[str, Any]) -> AgentGroup:
    agents = agent_group_config["agent_list"]
    text_model_configs = agent_group_config["model_configs"]
    model_configs = {}
    feature_extractor_configs = {}
    for model_id, conf in text_model_configs.items():
        feature_extractor_configs[model_id] = ModelConfig(**conf['feature_extractor'])
        model_configs[model_id] = ModelConfig(**conf['model'])
    optimizer_config = agent_group_config["optimizer"]
    optimizer_config = OptimizerConfig(**optimizer_config)
    return QMIXAgentGroup(agents, model_configs, feature_extractor_configs, optimizer_config)

def get_msg_aggr_agent_group(agent_group_config: Dict[str, Any]) -> AgentGroup:
    agents = agent_group_config["agent_list"]
    text_model_configs = agent_group_config["model_configs"]
    encoder_configs = {}
    feature_extractor_configs = {}
    decoder_configs = {}
    for model_id, conf in text_model_configs.items():
        feature_extractor_configs[model_id] = ModelConfig(**conf['feature_extractor'])
        encoder_configs[model_id] = ModelConfig(**conf['encoder'])
        decoder_configs[model_id] = ModelConfig(**conf['decoder'])
    aggr_model_config = ModelConfig(**agent_group_config["aggr_model_config"])
    optimizer_config = OptimizerConfig(**agent_group_config["optimizer"])
    return MsgAggrAgentGroup(
                        agents,
                        feature_extractor_configs,
                        encoder_configs,
                        decoder_configs,
                        aggr_model_config,
                        optimizer_config)

def get_gnn_agent_group(agent_group_config: Dict[str, Any]) -> AgentGroup:
    agents = agent_group_config["agent_list"]
    text_model_configs = agent_group_config["model_configs"]
    encoder_configs = {}
    feature_extractor_configs = {}
    decoder_configs = {}
    for model_id, conf in text_model_configs.items():
        feature_extractor_configs[model_id] = ModelConfig(**conf['feature_extractor'])
        encoder_configs[model_id] = ModelConfig(**conf['encoder'])
        decoder_configs[model_id] = ModelConfig(**conf['decoder'])
    graph_model_config = ModelConfig(**agent_group_config["graph_model_config"])
    graph_builder_config = GraphBuilderConfig(**agent_group_config["graph_builder_config"])
    optimizer_config = OptimizerConfig(**agent_group_config["optimizer"])
    return GNNAgentGroup(
                        agents,
                        feature_extractor_configs,
                        encoder_configs,
                        decoder_configs,
                        graph_builder_config,
                        graph_model_config,
                        optimizer_config)

def get_gnn_obs_comm_agent_group(agent_group_config: Dict[str, Any]) -> AgentGroup:
    agents = agent_group_config["agent_list"]
    text_model_configs = agent_group_config["model_configs"]
    encoder_configs = {}
    feature_extractor_configs = {}
    decoder_configs = {}
    for model_id, conf in text_model_configs.items():
        feature_extractor_configs[model_id] = ModelConfig(**conf['feature_extractor'])
        encoder_configs[model_id] = ModelConfig(**conf['encoder'])
        decoder_configs[model_id] = ModelConfig(**conf['decoder'])
    graph_model_config = ModelConfig(**agent_group_config["graph_model_config"])
    graph_builder_config = GraphBuilderConfig(**agent_group_config["graph_builder_config"])
    optimizer_config = OptimizerConfig(**agent_group_config["optimizer"])
    return GNNObsCommAgentGroup(
                        agents,
                        feature_extractor_configs,
                        encoder_configs,
                        decoder_configs,
                        graph_builder_config,
                        graph_model_config,
                        optimizer_config)

def get_random_agent_group(agent_group_config: Dict[str, Any]) -> AgentGroup:
    agents = agent_group_config["agent_list"]
    return RandomAgentGroup(agents)

def get_magent_prey_agent_group(agent_group_config: Dict[str, Any]) -> AgentGroup:
    agents = agent_group_config["agent_list"]
    return MagentPreyAgentGroup(agents)

def get_magent_battle_agent_group(agent_group_config: Dict[str, Any]) -> AgentGroup:
    agents = agent_group_config["agent_list"]
    return MagentBattleAgentGroup(agents)

registered_agent_groups = {
    "QMIX": get_qmix_agent_group,
    "MsgAggr": get_msg_aggr_agent_group,
    "GNN": get_gnn_agent_group,
    "GNNObsComm": get_gnn_obs_comm_agent_group,
    "Random": get_random_agent_group,
    "MagentPrey": get_magent_prey_agent_group,
    "MagentBattle": get_magent_battle_agent_group
}

class AgentGroupConfig(object):
    def __init__(self, **kwargs):
        self.agent_group_config = deepcopy(kwargs)
        self.ag_type = self.agent_group_config.pop("type")
        if self.ag_type not in registered_agent_groups:
            raise ValueError(f"Agent group type {self.ag_type} not registered.")

    def get_agent_group(self) -> AgentGroup:
        return registered_agent_groups[self.ag_type](self.agent_group_config)