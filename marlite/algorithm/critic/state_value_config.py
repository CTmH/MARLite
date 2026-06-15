from copy import deepcopy
from typing import Dict, Any, Callable
import torch.nn as nn

from marlite.algorithm.model.model_config import ModelConfig
from marlite.algorithm.critic.state_value import StateValue
from marlite.algorithm.critic.seq_state_value import SeqStateValue


def create_state_value(cfg):
    cfg = deepcopy(cfg)
    base_model_config = ModelConfig(**cfg.pop("base_model"))
    fe_config_dict = cfg.pop("feature_extractor", {"model_type": "Identity"})
    fe_config = ModelConfig(**fe_config_dict)
    return StateValue(base_model_config, fe_config)


def create_seq_state_value(cfg):
    cfg = deepcopy(cfg)
    base_model_config = ModelConfig(**cfg.pop("base_model"))
    fe_config_dict = cfg.pop("feature_extractor", {"model_type": "Identity"})
    fe_config = ModelConfig(**fe_config_dict)
    seq_model_config = ModelConfig(**cfg.pop("seq_model"))
    return SeqStateValue(base_model_config, fe_config, seq_model_config)


registered_state_value_creators: Dict[str, Callable[[Dict[str, Any]], nn.Module]] = {
    "StateValue": create_state_value,
    "SeqStateValue": create_seq_state_value,
}


class StateValueConfig:
    def __init__(self, **kwargs):
        cfg = deepcopy(kwargs)
        self.v_type = cfg.pop("type", "StateValue")
        if self.v_type not in registered_state_value_creators:
            raise ValueError(f"State value type '{self.v_type}' is not supported.")
        self.cfg = cfg

    def get_v_net(self):
        return registered_state_value_creators[self.v_type](deepcopy(self.cfg))
