from copy import deepcopy
from marlite.util.self_supervised_data_constructor.self_supervised_data_constructor import SelfSupervisedDataConstructor
from marlite.util.self_supervised_data_constructor.magent_vec_obs_data_constructor import MagentVecObsDataConstructor

registered_self_supervised_data_constructor_models = {
    "MagentVecObs": MagentVecObsDataConstructor,
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