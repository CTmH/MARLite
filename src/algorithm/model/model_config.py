from .rnn import RNNModel
from .custom_model import CustomModel
import torch.nn as nn

REGISTERED_MODELS = {
    "RNN": RNNModel,
    "Identity": nn.Identity,
    "Custom": CustomModel,
}

class ModelConfig:
    def __init__(self, **kwargs):
        self.model_type = kwargs.get("model_type")
        if self.model_type not in REGISTERED_MODELS:
            raise ValueError(f"Model type {self.model_type} not registered.")
        if self.model_type == "RNN":
            self.input_shape = kwargs.get("input_shape")
            self.output_shape = kwargs.get("output_shape")
            self.rnn_hidden_dim = kwargs.get("rnn_hidden_dim")
            self.rnn_layers = kwargs.get("rnn_layers", 1) # Default to 1 layer if not specified

    def __str__(self):
        discr = "{\n"
        for key, value in self.__dict__.items():
            discr += f"{key}: {value}, \n"
        return discr + "}"
    
    def get_model(self):
        if self.model_type == "RNN":
            model = RNNModel(self.input_shape, self.output_shape, self.rnn_hidden_dim, self.rnn_layers)
        elif self.model_type == "Identity":
            model = nn.Identity()
        else:
            raise ValueError(f"Model type {self.model_type} not registered.")
        return model
