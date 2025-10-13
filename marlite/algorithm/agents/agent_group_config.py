from copy import deepcopy
from typing import Dict, Any
from marlite.algorithm.agents.agent_group import AgentGroup
from marlite.algorithm.agents.qmix_agent_group import QMIXAgentGroup
from marlite.algorithm.agents.gnn_agent_group import GNNAgentGroup
from marlite.algorithm.agents.random_agent_group import RandomAgentGroup
from marlite.algorithm.agents.magent_agent_group import MAgentPreyAgentGroup, MAgentBattleAgentGroup
from marlite.algorithm.agents.msg_aggr_agent_group import MsgAggrAgentGroup, SeqMsgAggrAgentGroup
from marlite.algorithm.agents.gnn_obs_comm_agent_group import GNNObsCommAgentGroup
from marlite.algorithm.agents.g2anet_agent_group import G2ANetAgentGroup
from marlite.algorithm.model import ModelConfig
from marlite.algorithm.graph_builder import GraphBuilderConfig
from marlite.util.optimizer_config import OptimizerConfig
from marlite.util.lr_scheduler_config import LRSchedulerConfig

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
    lr_scheduler_config = agent_group_config.get("lr_scheduler", None)
    if lr_scheduler_config:
        lr_scheduler_config = LRSchedulerConfig(**lr_scheduler_config)
    return QMIXAgentGroup(agents, model_configs, feature_extractor_configs, optimizer_config, lr_scheduler_config)

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
    lr_scheduler_config = agent_group_config.get("lr_scheduler", None)
    if lr_scheduler_config:
        lr_scheduler_config = LRSchedulerConfig(**lr_scheduler_config)
    return MsgAggrAgentGroup(
                        agents,
                        feature_extractor_configs,
                        encoder_configs,
                        decoder_configs,
                        aggr_model_config,
                        optimizer_config,
                        lr_scheduler_config)

def get_seq_msg_aggr_agent_group(agent_group_config: Dict[str, Any]) -> AgentGroup:
    agents = agent_group_config["agent_list"]
    text_model_configs = agent_group_config["model_configs"]

    feature_extractor_configs = {}
    encoder_configs = {}
    decoder_configs = {}

    for model_id, conf in text_model_configs.items():
        feature_extractor_configs[model_id] = ModelConfig(**conf['feature_extractor'])
        encoder_configs[model_id] = ModelConfig(**conf['encoder'])
        decoder_configs[model_id] = ModelConfig(**conf['decoder'])

    aggr_model_config = ModelConfig(**agent_group_config["aggr_model_config"])
    optimizer_config = OptimizerConfig(**agent_group_config["optimizer"])

    lr_scheduler_config = agent_group_config.get("lr_scheduler", None)
    if lr_scheduler_config:
        lr_scheduler_config = LRSchedulerConfig(**lr_scheduler_config)

    return SeqMsgAggrAgentGroup(
        agents,
        feature_extractor_configs,
        encoder_configs,
        decoder_configs,
        aggr_model_config,
        optimizer_config,
        lr_scheduler_config
    )

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
    lr_scheduler_config = agent_group_config.get("lr_scheduler", None)
    if lr_scheduler_config:
        lr_scheduler_config = LRSchedulerConfig(**lr_scheduler_config)
    return GNNAgentGroup(
                        agents,
                        feature_extractor_configs,
                        encoder_configs,
                        decoder_configs,
                        graph_builder_config,
                        graph_model_config,
                        optimizer_config,
                        lr_scheduler_config)

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
    lr_scheduler_config = agent_group_config.get("lr_scheduler", None)
    if lr_scheduler_config:
        lr_scheduler_config = LRSchedulerConfig(**lr_scheduler_config)
    return GNNObsCommAgentGroup(
                        agents,
                        feature_extractor_configs,
                        encoder_configs,
                        decoder_configs,
                        graph_builder_config,
                        graph_model_config,
                        optimizer_config,
                        lr_scheduler_config)

def get_g2anet_agent_group(agent_group_config: Dict[str, Any]) -> AgentGroup:
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
    lr_scheduler_config = agent_group_config.get("lr_scheduler", None)
    if lr_scheduler_config:
        lr_scheduler_config = LRSchedulerConfig(**lr_scheduler_config)
    return G2ANetAgentGroup(
                        agents,
                        feature_extractor_configs,
                        encoder_configs,
                        decoder_configs,
                        graph_builder_config,
                        graph_model_config,
                        optimizer_config,
                        lr_scheduler_config)

def get_random_agent_group(agent_group_config: Dict[str, Any]) -> AgentGroup:
    agents = agent_group_config["agent_list"]
    return RandomAgentGroup(agents)

def get_magent_prey_agent_group(agent_group_config: Dict[str, Any]) -> AgentGroup:
    agents = agent_group_config["agent_list"]
    return MAgentPreyAgentGroup(agents)

def get_magent_battle_agent_group(agent_group_config: Dict[str, Any]) -> AgentGroup:
    agents = agent_group_config["agent_list"]
    return MAgentBattleAgentGroup(agents)

registered_agent_groups = {
    "QMIX": get_qmix_agent_group,
    "MsgAggr": get_msg_aggr_agent_group,
    "SeqMsgAggr": get_seq_msg_aggr_agent_group,
    "GNN": get_gnn_agent_group,
    "GNNObsComm": get_gnn_obs_comm_agent_group,
    "G2ANet": get_g2anet_agent_group,
    "Random": get_random_agent_group,
    "MAgentPrey": get_magent_prey_agent_group,
    "MAgentBattle": get_magent_battle_agent_group
}

class AgentGroupConfig(object):
    def __init__(self, **kwargs):
        self.agent_group_config = deepcopy(kwargs)
        self.ag_type = self.agent_group_config.pop("type")
        if self.ag_type not in registered_agent_groups:
            raise ValueError(f"Agent group type {self.ag_type} not registered.")

    def get_agent_group(self) -> AgentGroup:
        return registered_agent_groups[self.ag_type](self.agent_group_config)