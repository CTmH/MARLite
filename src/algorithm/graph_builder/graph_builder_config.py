from copy import deepcopy
from .graph_builder import GraphBuilder
from .magent_graph_builder import MagentGraphBuilder
from .partial_graph_builder import PartialGraphMagentBuilder

registered_graph_builder_models = {
    "Magent": MagentGraphBuilder,
    "PartialMagent": PartialGraphMagentBuilder
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