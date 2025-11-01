from copy import deepcopy
from marlite.algorithm.graph_builder.graph_builder import GraphBuilder
from marlite.algorithm.graph_builder.fullconn_graph_builder import FullConnGraphBuilder
from marlite.algorithm.graph_builder.magent_graph_builder import MAgentGraphBuilder
from marlite.algorithm.graph_builder.partial_graph_builder import PartialGraphMAgentBuilder, PartialGraphVectorStateBuilder
from marlite.algorithm.graph_builder.random_group_graph_builder import RandomGroupGraphBuilder
from marlite.algorithm.graph_builder.g2anet_graph_builder import G2ANetGraphBuilder

registered_graph_builder_models = {
    "FullConn": FullConnGraphBuilder,
    "RandomGroup": RandomGroupGraphBuilder,
    "MAgent": MAgentGraphBuilder,
    "PartialMAgent": PartialGraphMAgentBuilder,
    "PartialVectorState": PartialGraphVectorStateBuilder,
    "G2ANet": G2ANetGraphBuilder,
}
class GraphBuilderConfig:

    def __init__(self, **kwargs) -> None:
        self.conf = deepcopy(kwargs)
        self.builder_type = self.conf.pop("type")
        if self.builder_type not in registered_graph_builder_models:
            raise ValueError(f"Graph Builder type {self.builder_type} not registered.")
        self.graph_builder_class = registered_graph_builder_models[self.builder_type]

    def get_graph_builder(self) -> GraphBuilder:
        graph_builder = self.graph_builder_class(**self.conf)
        return graph_builder