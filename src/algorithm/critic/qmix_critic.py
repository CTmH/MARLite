import torch.nn as nn
import torch
import torch.nn.functional as F


class QMIXCritic(nn.Module):
    def __init__(self, state_shape, n_agents, qmix_hidden_dim, hyper_hidden_dim):
        super(QMIXCritic, self).__init__()
        self.state_shape = state_shape
        self.n_agents = n_agents
        self.qmix_hidden_dim = qmix_hidden_dim
        self.hyper_hidden_dim = hyper_hidden_dim

        self.hyper_w1 = nn.Linear(state_shape, n_agents * qmix_hidden_dim)
        self.hyper_w2 = nn.Linear(state_shape, qmix_hidden_dim * 1)

        self.hyper_b1 = nn.Linear(state_shape, qmix_hidden_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(state_shape, qmix_hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(qmix_hidden_dim, 1))

    # Need to check if algorithm is correct
    def forward(self, q_values, states):
        episode_num = q_values.size(0)
        q_values = q_values.view(-1, 1, self.n_agents)
        states = states.reshape(-1, self.state_shape)

        w1 = torch.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)

        w1 = w1.view(-1, self.n_agents, self.qmix_hidden_dim)
        b1 = b1.view(-1, 1, self.qmix_hidden_dim)

        hidden = F.elu(torch.bmm(q_values, w1) + b1)

        w2 = torch.abs(self.hyper_w2(states))
        b2 = self.hyper_b2(states)

        w2 = w2.view(-1, self.qmix_hidden_dim, 1)
        b2 = b2.view(-1, 1, 1)

        q_total = torch.bmm(hidden, w2) + b2
        q_total = q_total.view(episode_num, -1, 1)
        return q_total
    

class QMixer2HyperLayer(QMIXCritic):
    def __init__(self, state_shape, n_agents, qmix_hidden_dim, hyper_hidden_dim):
        super(QMIXCritic, self).__init__(state_shape, n_agents, qmix_hidden_dim, hyper_hidden_dim)

        self.hyper_w1 = nn.Sequential(nn.Linear(state_shape, hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hyper_hidden_dim, n_agents * qmix_hidden_dim))
        self.hyper_w2 = nn.Sequential(nn.Linear(state_shape, hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hyper_hidden_dim, qmix_hidden_dim))