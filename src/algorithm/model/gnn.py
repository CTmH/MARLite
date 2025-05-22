import torch.nn.functional as F
from torch import nn, zeros, Tensor
from torch_geometric.nn import GCNConv

class GCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim, add_self_loops=True)
        self.conv2 = GCNConv(hidden_dim, output_dim, add_self_loops=True)

    def forward(self, inputs: Tensor, edge_index):
        x = self.conv1(inputs, edge_index)
        x = F.leaky_relu(x)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        return x