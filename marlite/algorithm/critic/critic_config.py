# marlite/algorithm/critic/critic_config.py
from copy import deepcopy
from typing import Dict, Any, Callable
from torch.nn import Module

from marlite.algorithm.critic.mixer import QMixer, SeqQMixer, ProbQMixer, ProbSeqQMixer
from marlite.algorithm.model import ModelConfig


def create_qmixer(critic_config: Dict[str, Any]) -> QMixer:
    """Create a QMixer critic instance from config."""
    model_config = ModelConfig(**critic_config["model"])
    fe_config_dict = critic_config.get("feature_extractor", {"model_type": "Identity"})
    fe_config = ModelConfig(**fe_config_dict)
    return QMixer(model_config, fe_config)


def create_seq_qmixer(critic_config: Dict[str, Any]) -> SeqQMixer:
    """Create a SeqQMixer critic instance from config."""
    model_config = ModelConfig(**critic_config["model"])
    fe_config_dict = critic_config.get("feature_extractor", {"model_type": "Identity"})
    fe_config = ModelConfig(**fe_config_dict)

    seq_model_config = ModelConfig(**critic_config["seq_model"])
    state_feature_type = critic_config.get("state_feature_type", "Seq")

    return SeqQMixer(model_config, fe_config, seq_model_config, state_feature_type)

def create_prob_qmixer(critic_config: Dict[str, Any]) -> QMixer:
    """Create a QMixer critic instance from config."""
    model_config = ModelConfig(**critic_config.pop("model"))
    fe_config_dict = critic_config.pop("feature_extractor", {"model_type": "Identity"})
    fe_config = ModelConfig(**fe_config_dict)
    return ProbQMixer(model_config, fe_config, **critic_config)


def create_prob_seq_qmixer(critic_config: Dict[str, Any]) -> SeqQMixer:
    """Create a SeqQMixer critic instance from config."""
    model_config = ModelConfig(**critic_config.pop("model"))
    fe_config_dict = critic_config.pop("feature_extractor", {"model_type": "Identity"})
    fe_config = ModelConfig(**fe_config_dict)

    seq_model_config = ModelConfig(**critic_config.pop("seq_model"))

    return ProbSeqQMixer(model_config, fe_config, seq_model_config, **critic_config)

# Registry mapping critic type names to creator functions
registered_critic_creators: Dict[str, Callable[[Dict[str, Any]], Module]] = {
    "QMixer": create_qmixer,
    "SeqQMixer": create_seq_qmixer,
    "ProbQMixer": create_prob_qmixer,
    "ProbSeqQMixer": create_prob_seq_qmixer,
}


class CriticConfig:
    """
    Configuration class for creating critics in MARLite.

    Handles dispatching to the appropriate critic constructor based on type.
    """

    def __init__(self, **kwargs) -> None:
        self.critic_config: Dict[str, Any] = deepcopy(kwargs)
        self.critic_type: str = self.critic_config.pop("type", "QMixer")

        if self.critic_type not in registered_critic_creators:
            raise ValueError(f"Critic type {self.critic_type} is not supported.")

    def get_critic(self):
        """Instantiate and return the configured critic."""
        creator_fn = registered_critic_creators[self.critic_type]
        critic_config = deepcopy(self.critic_config)
        return creator_fn(critic_config)