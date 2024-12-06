import torch.nn as nn
import torch.nn.functional as F
from torch import zeros

class RNNModel(nn.Module):
    def __init__(self, layers: dict):
        super(RNNModel, self).__init__()
        input_shape, rnn_hidden_dim, output_shape = layers['input_shape'], layers['rnn_hidden_dim'], layers['output_shape']
        self.layers = layers
        self.rnn_layers = 1
        self.fc1 = nn.Linear(input_shape, rnn_hidden_dim)
        self.rnn = nn.GRU(rnn_hidden_dim, rnn_hidden_dim, num_layers=self.rnn_layers, batch_first=True)
        self.fc2 = nn.Linear(rnn_hidden_dim, output_shape)

    def init_hidden(self):
        # make hidden states on same device as model
        return zeros(self.rnn_layers, self.layers['rnn_hidden_dim'], device=self.fc1.weight.device)

    def forward(self, inputs, hidden_state):
        #print(inputs.shape)
        x = F.relu(self.fc1(inputs))
        #print(hidden_state.shape)
        h_in = hidden_state.reshape(-1, x.size(0), self.layers['rnn_hidden_dim'])
        #print(h_in.shape)
        #print(x.size())
        out, h = self.rnn(x, h_in)
        q = self.fc2(out)
        return q, h
