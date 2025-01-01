import torch.nn as nn
import torch

class Critic(nn.Module):
    def __init__(self, critic_model: nn.Module, feature_extractor: nn.Module):
        super(Critic, self).__init__()
        self.critic_model = critic_model
        self.feature_extractor = feature_extractor

    def forward(self, info_from_agents: torch.Tensor, states: torch.Tensor):
        state_features = self.feature_extractor(states)
        return self.critic_model(info_from_agents, state_features)
