import os
import yaml
import torch
import datetime
import numpy as np
from absl import logging

from marlite.trainer.trainer import Trainer
from marlite.util.serialization import (
    serialize_to_buffer,
    deserialize_from_buffer,
    get_state_dict,
    load_state_dict_into,
)


class OnPolicyTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_multi_gpu()
        self._compile_eval_models()

    def save_best_model(self):
        self.save_current_model(checkpoint="best")
        return self

    def train(self, **kwargs):
        raise NotImplementedError
