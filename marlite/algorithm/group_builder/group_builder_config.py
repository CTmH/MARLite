from copy import deepcopy
from marlite.algorithm.group_builder.group_builder import GroupBuilder
from marlite.algorithm.group_builder.magent_group_builder import (
    MAgentLabelPropagationGroupBuilder,
    MAgentVecLPGroupBuilder,
    MagentKMeansGroupBuilder,
    MagentVecKMeansGroupBuilder,
)
from marlite.algorithm.group_builder.partial_group_builder import (
    PartialGroupMAgentBuilder,
    PartialGroupVectorStateBuilder,
)
from marlite.algorithm.group_builder.fixed_group_builder import FixedGroupBuilder

registered_group_builders = {
    "MAgentLabelPropagation": MAgentLabelPropagationGroupBuilder,
    "MAgentVecLP": MAgentVecLPGroupBuilder,
    "MagentKMeans": MagentKMeansGroupBuilder,
    "MagentVecKMeans": MagentVecKMeansGroupBuilder,
    "PartialMAgent": PartialGroupMAgentBuilder,
    "PartialVectorState": PartialGroupVectorStateBuilder,
    "Fixed": FixedGroupBuilder,
}


class GroupBuilderConfig:
    def __init__(self, **kwargs) -> None:
        self.conf = deepcopy(kwargs)
        self.builder_type = self.conf.pop("type")
        if self.builder_type not in registered_group_builders:
            raise ValueError(
                f"Group Builder type {self.builder_type} not registered."
            )
        self.group_builder_class = registered_group_builders[self.builder_type]

    def get_group_builder(self) -> GroupBuilder:
        group_builder = self.group_builder_class(**self.conf)
        return group_builder
