import torch.nn as nn
import torch.nn.functional as F
from torch import zeros
from .custom_model import CustomModel

class TimeSeqModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
class RNNModel(TimeSeqModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    def forward(self, inputs):
        batch_size = inputs.size(0)
        x = F.relu(self.fc1(inputs))
        h_in = zeros(self.rnn_layers, batch_size, self.rnn_dim).to(inputs.device)
        out, _ = self.rnn(x, h_in)
        q = self.fc2(out)
        return q[:,-1,:] # get the last output of the sequence

class CustomTimeSeqModel(TimeSeqModel, CustomModel):
    def __init__(self, **kwargs):
        super(CustomTimeSeqModel, self).__init__()
        super(CustomModel, self).__init__(**kwargs)

    def forward(self, inputs):
        return CustomModel.forward(self, inputs)