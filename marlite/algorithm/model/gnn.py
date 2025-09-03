import torch.nn.functional as F
from torch import nn, zeros, Tensor
from torch_geometric.nn import GCNConv, GATConv

class GCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation='ELU'):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim, add_self_loops=True)
        self.conv2 = GCNConv(hidden_dim, output_dim, add_self_loops=True)

        activation_nn_module = getattr(nn, activation, None)
        if activation_nn_module is None:
            raise ValueError(f"Invalid activation function: {activation}")
        self.activation = activation_nn_module()

    def forward(self, inputs: Tensor, edge_index):
        x = self.conv1(inputs, edge_index)
        x = self.activation(x)
        x = self.conv2(x, edge_index)
        x = self.activation(x)
        return x

class GATModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, head_conv1=8, head_conv2=1, dropout=0.75, activation='ELU'):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            heads=head_conv1,
            concat=True,
            dropout=dropout,
            add_self_loops=True,
        )
        self.conv2 = GATConv(
            in_channels=hidden_dim * head_conv1,
            out_channels=output_dim,
            heads=head_conv2,
            concat=False,
            dropout=dropout,
            add_self_loops=True,
        )

        activation_nn_module = getattr(nn, activation, None)
        if activation_nn_module is None:
            raise ValueError(f"Invalid activation function: {activation}")
        self.activation = activation_nn_module()

    def forward(self, inputs: Tensor, edge_index):
        x = self.conv1(inputs, edge_index)
        x = self.activation(x)
        x = self.conv2(x, edge_index)
        x = self.activation(x)
        return x