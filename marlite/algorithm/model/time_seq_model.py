import torch.nn as nn

class TimeSeqModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
