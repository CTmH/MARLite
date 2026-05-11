from copy import deepcopy
from marlite.util.self_supervised_data_constructor.self_supervised_data_constructor import SelfSupervisedDataConstructor
from marlite.util.self_supervised_data_constructor.magent_obs_data_constructor import MagentVecObsDataConstructor, MagentImageObsDataConstructor
from marlite.util.self_supervised_data_constructor.sumo_obs_data_constructor import SumoObsDataConstructor
from marlite.util.self_supervised_data_constructor.magent_group_window_constructor import MagentGroupWindowConstructor
from marlite.util.self_supervised_data_constructor.magent_group_features_constructor import MagentGroupFeaturesConstructor
from marlite.util.self_supervised_data_constructor.magent_vec_state_group_features_constructor import (
    MagentVecStateGroupFeaturesConstructor,
)

registered_self_supervised_data_constructor_models = {
    "MagentVecObs": MagentVecObsDataConstructor,
    "MagentImageObs": MagentImageObsDataConstructor,
    "SUMO": SumoObsDataConstructor,
    "MagentGroupWindow": MagentGroupWindowConstructor,
    "MagentGroupFeatures": MagentGroupFeaturesConstructor,
    "MagentVecStateGroupFeatures": MagentVecStateGroupFeaturesConstructor,
}

class SelfSupervisedDataConstructorConfig:

    def __init__(self, **kwargs) -> None:
        self.conf = deepcopy(kwargs)
        self.constructor_type = self.conf.pop("type")
        if self.constructor_type not in registered_self_supervised_data_constructor_models:
            raise ValueError(f"SelfSupervisedDataConstructor type {self.constructor_type} not registered.")
        self.constructor_class = registered_self_supervised_data_constructor_models[self.constructor_type]

    def get_data_constructor(self) -> SelfSupervisedDataConstructor:
        data_constructor = self.constructor_class(**self.conf)
        return data_constructor