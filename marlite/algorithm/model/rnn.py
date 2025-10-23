from marlite.algorithm.model.time_seq_model import TimeSeqModel
from torch import nn, zeros
from torch.functional import F

class RNNModel(TimeSeqModel):
    def __init__(self, *args, **kwargs):
        super().__init__()

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
        h_in = zeros(self.rnn_layers, batch_size, self.rnn_dim,
                    dtype=inputs.dtype, device=inputs.device)
        out, _ = self.rnn(x, h_in)
        q = self.fc2(out)
        return q[:,-1,:] # get the last output of the sequence