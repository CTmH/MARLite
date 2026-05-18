from copy import deepcopy
from typing import Dict, Any, Type
from marlite.algorithm.agents.agent_group import AgentGroup
from marlite.algorithm.agents.qmix_agent_group import QMIXAgentGroup
from marlite.algorithm.agents.gnn_agent_group import GNNAgentGroup
from marlite.algorithm.agents.random_agent_group import RandomAgentGroup
from marlite.algorithm.agents.magent_agent_group import (
    MAgentPreyAgentGroup,
    MAgentBattleAgentGroup,
)
from marlite.algorithm.agents.msg_aggr_agent_group import (
    ObsMsgAggrAgentGroup,
    SeqMsgAggrAgentGroup,
)
from marlite.algorithm.agents.msg_aggr_agent_group import (
    ProbObsMsgAggrAgentGroup,
    ProbSeqMsgAggrAgentGroup,
)
from marlite.algorithm.agents.msg_aggr_agent_group import DualPathObsMsgAggrAgentGroup
from marlite.algorithm.agents.msg_aggr_agent_group import (
    DualPathProbObsMsgAggrAgentGroup,
)
from marlite.algorithm.agents.gnn_comm_agent_group import (
    ObsGNNCommAgentGroup,
    SeqGNNCommAgentGroup,
)
from marlite.algorithm.agents.gnn_comm_agent_group import (
    ProbObsGNNCommAgentGroup,
    ProbSeqGNNCommAgentGroup,
)
from marlite.algorithm.agents.gnn_comm_agent_group import (
    DualPathObsGNNCommAgentGroup,
    DualPathProbObsGNNCommAgentGroup,
)
from marlite.algorithm.agents.g2anet_agent_group import G2ANetAgentGroup
from marlite.algorithm.agents.group_consensus_agent_group import GroupConsensusAgentGroup
from marlite.algorithm.agents.mappo_agent_group import MAPPOAgentGroup
from marlite.algorithm.agents.vaegc_mappo_agent_group import VAEGroupConsensusMAPPOAgentGroup
from marlite.algorithm.model import ModelConfig
from marlite.algorithm.graph_builder import GraphBuilderConfig
from marlite.algorithm.group_builder import GroupBuilderConfig


def create_qmix_agent_group(agent_group_config: Dict[str, Any]) -> AgentGroup:
    agents = agent_group_config.pop("agent_list")
    text_model_configs = agent_group_config.pop("model_configs")
    model_configs = {}
    feature_extractor_configs = {}
    for model_id, conf in text_model_configs.items():
        feature_extractor_configs[model_id] = ModelConfig(**conf["feature_extractor"])
        model_configs[model_id] = ModelConfig(**conf["model"])
    return QMIXAgentGroup(
        agents, model_configs, feature_extractor_configs, **agent_group_config
    )


def create_gnn_agent_group(agent_group_config: Dict[str, Any]) -> AgentGroup:
    return _create_gnn_agent_group(GNNAgentGroup, agent_group_config)


def create_obs_gnn_comm_agent_group(agent_group_config: Dict[str, Any]) -> AgentGroup:
    return _create_gnn_agent_group(ObsGNNCommAgentGroup, agent_group_config)


def create_seq_gnn_comm_agent_group(agent_group_config: Dict[str, Any]) -> AgentGroup:
    return _create_gnn_agent_group(SeqGNNCommAgentGroup, agent_group_config)


def create_prob_obs_gnn_comm_agent_group(
    agent_group_config: Dict[str, Any],
) -> AgentGroup:
    return _create_gnn_agent_group(ProbObsGNNCommAgentGroup, agent_group_config)


def create_prob_seq_gnn_comm_agent_group(
    agent_group_config: Dict[str, Any],
) -> AgentGroup:
    return _create_gnn_agent_group(ProbSeqGNNCommAgentGroup, agent_group_config)


def create_dual_path_obs_gnn_comm_agent_group(
    agent_group_config: Dict[str, Any],
) -> AgentGroup:
    return _create_dual_path_gnn_agent_group(
        DualPathObsGNNCommAgentGroup, agent_group_config
    )


def create_dual_path_prob_obs_gnn_comm_agent_group(
    agent_group_config: Dict[str, Any],
) -> AgentGroup:
    return _create_dual_path_gnn_agent_group(
        DualPathProbObsGNNCommAgentGroup, agent_group_config
    )


def create_g2anet_agent_group(agent_group_config: Dict[str, Any]) -> AgentGroup:
    return _create_gnn_agent_group(G2ANetAgentGroup, agent_group_config)


def _create_gnn_agent_group(
    agent_group_class: Type[AgentGroup], agent_group_config: Dict[str, Any]
) -> AgentGroup:
    agents = agent_group_config.pop("agent_list")
    text_model_configs = agent_group_config.pop("model_configs")
    encoder_configs = {}
    feature_extractor_configs = {}
    decoder_configs = {}
    for model_id, conf in text_model_configs.items():
        feature_extractor_configs[model_id] = ModelConfig(**conf["feature_extractor"])
        encoder_configs[model_id] = ModelConfig(**conf["encoder"])
        decoder_configs[model_id] = ModelConfig(**conf["decoder"])
    graph_model_config = ModelConfig(**agent_group_config.pop("graph_model_config"))
    graph_builder_config = GraphBuilderConfig(
        **agent_group_config.pop("graph_builder")
    )
    return agent_group_class(
        agents,
        feature_extractor_configs,
        encoder_configs,
        decoder_configs,
        graph_builder_config,
        graph_model_config,
        **agent_group_config,
    )


def create_obs_msg_aggr_agent_group(agent_group_config: Dict[str, Any]) -> AgentGroup:
    return _create_msg_agent_group(ObsMsgAggrAgentGroup, agent_group_config)


def create_seq_msg_aggr_agent_group(agent_group_config: Dict[str, Any]) -> AgentGroup:
    return _create_msg_agent_group(SeqMsgAggrAgentGroup, agent_group_config)


def create_prob_obs_msg_aggr_agent_group(
    agent_group_config: Dict[str, Any],
) -> AgentGroup:
    return _create_msg_agent_group(ProbObsMsgAggrAgentGroup, agent_group_config)


def create_prob_seq_msg_aggr_agent_group(
    agent_group_config: Dict[str, Any],
) -> AgentGroup:
    return _create_msg_agent_group(ProbSeqMsgAggrAgentGroup, agent_group_config)


def create_dual_path_obs_msg_aggr_agent_group(
    agent_group_config: Dict[str, Any],
) -> AgentGroup:
    return _create_dual_path_msg_agent_group(
        DualPathObsMsgAggrAgentGroup, agent_group_config
    )


"""
def create_dual_path_seq_msg_aggr_agent_group(agent_group_config: Dict[str, Any]) -> AgentGroup:
    return _create_dual_path_msg_agent_group(DualPathSeqMsgAggrAgentGroup, agent_group_config)
"""


def create_dual_path_prob_obs_msg_aggr_agent_group(
    agent_group_config: Dict[str, Any],
) -> AgentGroup:
    return _create_dual_path_msg_agent_group(
        DualPathProbObsMsgAggrAgentGroup, agent_group_config
    )


"""
def create_dual_path_prob_seq_msg_aggr_agent_group(agent_group_config: Dict[str, Any]) -> AgentGroup:
    return _create_dual_path_msg_agent_group(DualPathProbSeqMsgAggrAgentGroup, agent_group_config)
"""


def _create_msg_agent_group(
    agent_group_class: Type[AgentGroup], agent_group_config: Dict[str, Any]
) -> AgentGroup:
    agents = agent_group_config.pop("agent_list")
    text_model_configs = agent_group_config.pop("model_configs")

    feature_extractor_configs = {}
    encoder_configs = {}
    decoder_configs = {}

    for model_id, conf in text_model_configs.items():
        feature_extractor_configs[model_id] = ModelConfig(**conf["feature_extractor"])
        encoder_configs[model_id] = ModelConfig(**conf["encoder"])
        decoder_configs[model_id] = ModelConfig(**conf["decoder"])

    aggr_model_config = ModelConfig(**agent_group_config.pop("aggr_model_config"))

    return agent_group_class(
        agent_model_dict=agents,
        feature_extractor_configs=feature_extractor_configs,
        encoder_configs=encoder_configs,
        decoder_configs=decoder_configs,
        aggr_model_config=aggr_model_config,
        **agent_group_config,
    )


def _create_dual_path_msg_agent_group(
    agent_group_class: Type[AgentGroup], agent_group_config: Dict[str, Any]
) -> AgentGroup:
    agents = agent_group_config.pop("agent_list")
    text_model_configs = agent_group_config.pop("model_configs")

    feature_extractor_configs = {}
    msg_feature_extractor_configs = {}
    encoder_configs = {}
    decoder_configs = {}

    for model_id, conf in text_model_configs.items():
        feature_extractor_configs[model_id] = ModelConfig(**conf["feature_extractor"])
        msg_feature_extractor_configs[model_id] = ModelConfig(
            **conf["msg_feature_extractor"]
        )
        encoder_configs[model_id] = ModelConfig(**conf["encoder"])
        decoder_configs[model_id] = ModelConfig(**conf["decoder"])

    aggr_model_config = ModelConfig(**agent_group_config.pop("aggr_model_config"))

    return agent_group_class(
        agent_model_dict=agents,
        feature_extractor_configs=feature_extractor_configs,
        msg_feature_extractor_configs=msg_feature_extractor_configs,
        encoder_configs=encoder_configs,
        decoder_configs=decoder_configs,
        aggr_model_config=aggr_model_config,
        **agent_group_config,
    )


def _create_dual_path_gnn_agent_group(
    agent_group_class: Type[AgentGroup], agent_group_config: Dict[str, Any]
) -> AgentGroup:
    agents = agent_group_config.pop("agent_list")
    text_model_configs = agent_group_config.pop("model_configs")

    feature_extractor_configs = {}
    msg_feature_extractor_configs = {}
    encoder_configs = {}
    decoder_configs = {}

    for model_id, conf in text_model_configs.items():
        feature_extractor_configs[model_id] = ModelConfig(**conf["feature_extractor"])
        msg_feature_extractor_configs[model_id] = ModelConfig(
            **conf["msg_feature_extractor"]
        )
        encoder_configs[model_id] = ModelConfig(**conf["encoder"])
        decoder_configs[model_id] = ModelConfig(**conf["decoder"])

    graph_model_config = ModelConfig(**agent_group_config.pop("graph_model_config"))
    graph_builder_config = GraphBuilderConfig(
        **agent_group_config.pop("graph_builder")
    )
    return agent_group_class(
        agent_model_dict=agents,
        feature_extractor_configs=feature_extractor_configs,
        msg_feature_extractor_configs=msg_feature_extractor_configs,
        encoder_configs=encoder_configs,
        decoder_configs=decoder_configs,
        graph_builder_config=graph_builder_config,
        graph_model_config=graph_model_config,
        **agent_group_config,
    )


def create_mappo_agent_group(agent_group_config: Dict[str, Any]) -> AgentGroup:
    agents = agent_group_config.pop("agent_list")
    text_model_configs = agent_group_config.pop("model_configs")
    model_configs = {}
    feature_extractor_configs = {}
    for model_id, conf in text_model_configs.items():
        feature_extractor_configs[model_id] = ModelConfig(**conf["feature_extractor"])
        model_configs[model_id] = ModelConfig(**conf["model"])
    return MAPPOAgentGroup(
        agents, model_configs, feature_extractor_configs, **agent_group_config
    )


def create_random_agent_group(agent_group_config: Dict[str, Any]) -> AgentGroup:
    agents = agent_group_config["agent_list"]
    return RandomAgentGroup(agents)


def create_magent_prey_agent_group(agent_group_config: Dict[str, Any]) -> AgentGroup:
    agents = agent_group_config["agent_list"]
    strategy = agent_group_config.get("strategy", "greedy")
    temperature = agent_group_config.get("temperature", 1.0)
    top_k = agent_group_config.get("top_k", 5)
    return MAgentPreyAgentGroup(agents, strategy, temperature, top_k)


def create_magent_battle_agent_group(agent_group_config: Dict[str, Any]) -> AgentGroup:
    agents = agent_group_config["agent_list"]
    strategy = agent_group_config.get("strategy", "advanced")
    temperature = agent_group_config.get("temperature", 1.0)
    top_k = agent_group_config.get("top_k", 8)
    return MAgentBattleAgentGroup(agents, strategy, temperature, top_k)


def create_group_consensus_agent_group(
    agent_group_config: Dict[str, Any]
) -> GroupConsensusAgentGroup:
    """Create a GroupConsensusAgentGroup from config."""
    return _create_group_consensus_agent_group(
        GroupConsensusAgentGroup, agent_group_config
    )


def create_vaegc_mappo_agent_group(
    agent_group_config: Dict[str, Any]
) -> VAEGroupConsensusMAPPOAgentGroup:
    """Create a VAEGroupConsensusMAPPOAgentGroup from config."""
    return _create_group_consensus_agent_group(
        VAEGroupConsensusMAPPOAgentGroup, agent_group_config
    )


def _create_group_consensus_agent_group(
    agent_group_class: Type[AgentGroup], agent_group_config: Dict[str, Any]
) -> AgentGroup:
    agents = agent_group_config.pop("agent_list")
    text_model_configs = agent_group_config.pop("model_configs")

    feature_extractor_configs = {}
    group_estimate_feature_extractor_configs = {}
    encoder_configs = {}
    decoder_configs = {}

    for model_id, conf in text_model_configs.items():
        feature_extractor_configs[model_id] = ModelConfig(**conf["feature_extractor"])
        group_estimate_feature_extractor_configs[model_id] = ModelConfig(
            **conf["group_estimate_feature_extractor"]
        )
        encoder_configs[model_id] = ModelConfig(**conf["encoder"])
        decoder_configs[model_id] = ModelConfig(**conf["decoder"])

    group_builder_config = GroupBuilderConfig(
        **agent_group_config.pop("group_builder")
    )

    return agent_group_class(
        agent_model_dict=agents,
        feature_extractor_configs=feature_extractor_configs,
        group_estimate_feature_extractor_configs=group_estimate_feature_extractor_configs,
        encoder_configs=encoder_configs,
        decoder_configs=decoder_configs,
        group_builder_config=group_builder_config,
        **agent_group_config,
    )


registered_agent_groups = {
    "QMIX": create_qmix_agent_group,
    "MAPPO": create_mappo_agent_group,
    "MsgAggr": create_obs_msg_aggr_agent_group,
    "ObsMsgAggr": create_obs_msg_aggr_agent_group,
    "SeqMsgAggr": create_seq_msg_aggr_agent_group,
    "ProbObsMsgAggr": create_prob_obs_msg_aggr_agent_group,
    "ProbSeqMsgAggr": create_prob_seq_msg_aggr_agent_group,
    "DualPathObsMsgAggr": create_dual_path_obs_msg_aggr_agent_group,
    # "DualPathSeqMsgAggr": create_dual_path_seq_msg_aggr_agent_group,
    "DualPathProbObsMsgAggr": create_dual_path_prob_obs_msg_aggr_agent_group,
    # "DualPathProbSeqMsgAggr": create_dual_path_prob_seq_msg_aggr_agent_group,
    "GNN": create_gnn_agent_group,
    "ObsGNNComm": create_obs_gnn_comm_agent_group,
    "SeqGNNComm": create_seq_gnn_comm_agent_group,
    "ProbObsGNNComm": create_prob_obs_gnn_comm_agent_group,
    "ProbSeqGNNComm": create_prob_seq_gnn_comm_agent_group,
    "DualPathObsGNNComm": create_dual_path_obs_gnn_comm_agent_group,
    "DualPathProbObsGNNComm": create_dual_path_prob_obs_gnn_comm_agent_group,
    "G2ANet": create_g2anet_agent_group,
    "Random": create_random_agent_group,
    "MAgentPrey": create_magent_prey_agent_group,
    "MAgentBattle": create_magent_battle_agent_group,
    "GroupConsensusQMIX": create_group_consensus_agent_group,
    "VAEGroupConsensusMAPPO": create_vaegc_mappo_agent_group,
}


class AgentGroupConfig(object):
    def __init__(self, **kwargs):
        self.agent_group_config = deepcopy(kwargs)
        self.ag_type = self.agent_group_config.pop("type")
        self.agent_group_config.pop("optimizer", None)
        self.agent_group_config.pop("lr_scheduler", None)
        if self.ag_type not in registered_agent_groups:
            raise ValueError(f"Agent group type {self.ag_type} not registered.")

    def get_agent_group(self) -> AgentGroup:
        return registered_agent_groups[self.ag_type](deepcopy(self.agent_group_config))
