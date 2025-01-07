import torch
import logging
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
        self.pretrained_params_path = kwargs.pop("pretrained_params_path", None)
        self.model_config = kwargs
        if self.model_type not in REGISTERED_MODELS:
            raise ValueError(f"Model type {self.model_type} not registered.")

    def __str__(self):
        discr = "{\n"
        for key, value in self.__dict__.items():
            discr += f"{key}: {value}, \n"
        return discr + "}"
    
    def get_model(self):
        if self.model_type in REGISTERED_MODELS:
            model_class = REGISTERED_MODELS[self.model_type]
            model = model_class(**self.model_config)
            if self.pretrained_params_path is not None:
                try:
                    model.load_state_dict(torch.load(self.pretrained_params_path, weights_only=True))
                except FileNotFoundError as e:
                    logging.error(f"Pretrained model path {self.pretrained_params_path} not found.")
                    raise e
        else:
            raise ValueError(f"Model type {self.model_type} not registered.")
        return model
