from marlite.algorithm.critic.critic import Critic, SeqCritic
from marlite.algorithm.model import ModelConfig
from marlite.algorithm.model.qmix_critic_model import QMIXCriticModel
from marlite.algorithm.model.graphmix_critic_model import GraphMIXCriticModel


registered_critic_models = {
    "QMIX": QMIXCriticModel,
    "GraphMIX": GraphMIXCriticModel,
}


class CriticConfig:
    def __init__(self, **kwargs):
        self.critic_type = kwargs.pop("type")
        self.state_feature_type = kwargs.pop("state_feature_type", 'Seq')
        if self.critic_type not in registered_critic_models:
            raise ValueError(f"Critic type {self.critic_type} not registered.")

        # Handle feature extractor configuration
        if "feature_extractor" in kwargs:
            fe_conf = kwargs.pop("feature_extractor")
        else:
            fe_conf = {"model_type": "Identity"}
        self.critic_fe_config = ModelConfig(**fe_conf)

        # Handle sequence model configuration
        if "seq_model" in kwargs:
            seq_model_conf = kwargs.pop("seq_model")
            self.critic_seq_model_conf = ModelConfig(**seq_model_conf)
            self.use_seq_model = True
        else:
            self.use_seq_model = False

        # Remaining model config parameters
        self.model_config = kwargs
        self.critic_model_class = registered_critic_models[self.critic_type]

    def get_critic(self):
        if self.use_seq_model:
            critic = SeqCritic(
                self.critic_model_class(**self.model_config),
                self.critic_fe_config.get_model(),
                self.critic_seq_model_conf.get_model(),
                self.state_feature_type
            )
        else:
            critic = Critic(
                self.critic_model_class(**self.model_config),
                self.critic_fe_config.get_model()
            )
        return critic