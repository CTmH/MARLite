import torch.nn as nn
import torch.nn.functional as F
from torch import zeros

class RNNModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_hidden(self):
        raise NotImplementedError

class GRUModel(RNNModel):
    def __init__(self, input_shape, output_shape, rnn_hidden_dim, rnn_layers=1):
        super(GRUModel, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.rnn_dim = rnn_hidden_dim
        self.rnn_layers = rnn_layers
        self.fc1 = nn.Linear(input_shape, rnn_hidden_dim)
        self.rnn = nn.GRU(rnn_hidden_dim, rnn_hidden_dim, num_layers=self.rnn_layers, batch_first=True)
        self.fc2 = nn.Linear(rnn_hidden_dim, output_shape)

    def init_hidden(self):
        # make hidden states on same device as model
        return zeros(self.rnn_layers, self.rnn_dim, device=self.fc1.weight.device)

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, x.size(0), self.rnn_dim)
        out, h = self.rnn(x, h_in)
        q = self.fc2(out)
        return q, h
