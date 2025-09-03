import torch.nn as nn
import torch
import torch.nn.functional as F

# TODO: add hidden layer configuration for the critic network. 
class QMIXCriticModel(nn.Module):
    def __init__(self, state_shape, input_dim, qmix_hidden_dim, hypernet_layers=2, hyper_hidden_dim=128):
        super(QMIXCriticModel, self).__init__()
        self.state_shape = state_shape
        self.input_dim = input_dim
        self.qmix_hidden_dim = qmix_hidden_dim
        self.hypernet_layers = hypernet_layers
        self.hyper_hidden_dim = hyper_hidden_dim

        if hypernet_layers == 1:
            self.hyper_w1 = nn.Linear(state_shape, input_dim * qmix_hidden_dim)
            self.hyper_w2 = nn.Linear(state_shape, qmix_hidden_dim * 1)
        else:
            self.hyper_w1 = nn.Sequential(nn.Linear(state_shape, hyper_hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(hyper_hidden_dim, input_dim * qmix_hidden_dim))
            self.hyper_w2 = nn.Sequential(nn.Linear(state_shape, hyper_hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(hyper_hidden_dim, qmix_hidden_dim * 1))

        self.hyper_b1 = nn.Linear(state_shape, qmix_hidden_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(state_shape, qmix_hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(qmix_hidden_dim, 1))

    # Need to check if algorithm is correct
    def forward(self, q_val: torch.Tensor, states: torch.Tensor):
        bs = q_val.size(0)
        q_val = q_val.reshape(bs, 1, self.input_dim) # (B, 1, N)
        w1 = torch.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)

        w1 = w1.view(bs, self.input_dim, self.qmix_hidden_dim)
        b1 = b1.view(bs, 1, self.qmix_hidden_dim)

        hidden = F.elu(torch.bmm(q_val, w1) + b1)

        w2 = torch.abs(self.hyper_w2(states))
        b2 = self.hyper_b2(states)

        w2 = w2.view(-1, self.qmix_hidden_dim, 1)
        b2 = b2.view(-1, 1, 1)

        q_tot = torch.bmm(hidden, w2) + b2
        q_tot = q_tot.view(bs)
        return q_tot