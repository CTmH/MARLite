from torch import nn

class GraphBuilder(nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, state):
        raise NotImplementedError
    
    def reset(self) -> nn.Module:
        raise self