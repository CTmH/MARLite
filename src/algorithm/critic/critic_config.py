from .critic import Critic
from ..model.model_config import ModelConfig
from .qmix_critic_model import QMIXCriticModel

registered_critic_models = {
    "QMIX": QMIXCriticModel,
}

class CriticConfig:
    def __init__(self, **kwargs):
        self.critic_type = kwargs.pop("type")
        if self.critic_type not in registered_critic_models:
            raise ValueError(f"Critic type {self.critic_type} not registered.")
        if 'feature_extractor' in kwargs:  # Check if feature extractor is defined in the critic configuration. If not, use Identity as default.
            fe_conf = kwargs.pop('feature_extractor')
        else:
            fe_conf = {'model_type': 'Identity'}
        critic_fe_config = ModelConfig(**fe_conf)
        self.critic_feature_extractor = critic_fe_config.get_model()
        model_config = kwargs
        critic_model = registered_critic_models[self.critic_type]
        self.critic_model = critic_model(**model_config)

    def get_critic(self):
        critic = Critic(self.critic_model, self.critic_feature_extractor)
        return critic