import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class HyperNetwork(nn.Module):
    def __init__(
        self,
        cond_dim: int,
        layer_dims: List[int],
        cond_hidden_dim: int = 128,
        cond_hidden_layers: int = 1,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims) - 1

        self.hyper_ws = nn.ModuleList()
        self.hyper_bs = nn.ModuleList()

        for i in range(self.num_layers):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i + 1]
            w_size = in_dim * out_dim

            if cond_hidden_layers == 1:
                hyper_w = nn.Linear(cond_dim, w_size)
                hyper_b = nn.Linear(cond_dim, out_dim)
            else:
                hyper_w = nn.Sequential(
                    nn.Linear(cond_dim, cond_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(cond_hidden_dim, w_size),
                )
                hyper_b = nn.Sequential(
                    nn.Linear(cond_dim, cond_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(cond_hidden_dim, out_dim),
                )

            self.hyper_ws.append(hyper_w)
            self.hyper_bs.append(hyper_b)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        h = x

        for i in range(self.num_layers):
            in_dim = self.layer_dims[i]
            out_dim = self.layer_dims[i + 1]

            w = torch.abs(self.hyper_ws[i](z))
            b = self.hyper_bs[i](z)

            w = w.view(bs, in_dim, out_dim)
            b = b.view(bs, 1, out_dim)

            h = h.unsqueeze(1)
            h = torch.bmm(h, w) + b
            h = h.squeeze(1)

            if i < self.num_layers - 1:
                h = F.elu(h)

        return h
