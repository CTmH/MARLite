from .qmix_critic import QMIXCritic

Registered_Critics = {
    "QMIX": QMIXCritic,
}

class CriticConfig:
    def __init__(self, **kwargs):
        self.critic_type = kwargs.pop("critic_type")
        if self.critic_type not in Registered_Critics:
            raise ValueError(f"Critic type {self.critic_type} not registered.")
        self.critic_config = kwargs

    def get_critic(self):
        critic_class = Registered_Critics[self.critic_type]
        return critic_class(**self.critic_config)