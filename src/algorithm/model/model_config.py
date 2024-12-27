from .rnn import RNNModel
from .custom_model import CustomModel
from .flatten import Flatten
import torch.nn as nn

REGISTERED_MODELS = {
    "RNN": RNNModel,
    "Identity": nn.Identity,
    "Flatten": Flatten,
    "Custom": CustomModel,
}

class ModelConfig:
    def __init__(self, **kwargs):
        self.model_type = kwargs.pop("model_type")
        self.model_config = kwargs
        if self.model_type not in REGISTERED_MODELS:
            raise ValueError(f"Model type {self.model_type} not registered.")

    def __str__(self):
        discr = "{\n"
        for key, value in self.__dict__.items():
            discr += f"{key}: {value}, \n"
        return discr + "}"
    
    def get_model(self):
        if self.model_type == "RNN":
            model = RNNModel(**self.model_config)
        elif self.model_type == "Identity":
            model = nn.Identity()
        elif self.model_type == "Flatten":
            model = Flatten()
        else:
            raise ValueError(f"Model type {self.model_type} not registered.")
        return model
