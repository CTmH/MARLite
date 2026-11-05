import torch.optim as optim
from torch import nn

registered_optimizers = {
    "Adam": optim.Adam,
    "SGD": optim.SGD,
    "RMSprop": optim.RMSprop,
}

class OptimizerConfig:
    def __init__(self, **kwargs):
        self.optimizer_type = kwargs.pop('type')
        self.optimizer_kwargs = kwargs

    def get_optimizer(self, params_dict) -> optim.Optimizer:
        if self.optimizer_type in registered_optimizers:
            optim_class = registered_optimizers[self.optimizer_type]
            if isinstance(params_dict, nn.Module):
                if hasattr(params_dict,"optimizer_parameter_groups"):
                    params_dict = params_dict.optimizer_parameter_groups(self.optimizer_kwargs.get("lr",.001))
                else:
                    params_dict = params_dict.parameters()
            return optim_class(params_dict, **self.optimizer_kwargs)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")