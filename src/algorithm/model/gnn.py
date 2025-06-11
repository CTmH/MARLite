import torch.nn.functional as F
from torch import nn, zeros, Tensor
from torch_geometric.nn import GCNConv, GATConv

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
    
class GATModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, head_conv1 = 8):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            heads=head_conv1,  # 使用8个注意力头
            concat=True,
            dropout=0.75,
            add_self_loops=True,
        )
        self.conv2 = GATConv(
            in_channels=hidden_dim * head_conv1,  # 因为concat=True时输出维度是hidden_dim * heads
            out_channels=output_dim,
            heads=1,  # 最后一层用单个注意力头
            concat=False,
            dropout=0.75,
            add_self_loops=True,
        )

    def forward(self, inputs: Tensor, edge_index):
        x = F.dropout(inputs, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x