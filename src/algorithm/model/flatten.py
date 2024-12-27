import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        #return x.view(x.size(0), -1)
        return torch.flatten(x, start_dim=1)