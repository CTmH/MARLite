from marlite.algorithm.model.time_seq_model import TimeSeqModel
from marlite.algorithm.model.custom_model import CustomModel

class Conv1DModel(TimeSeqModel):
    def __init__(self, *args, **kwargs):
        super().__init__()

class CustomConv1DModel(Conv1DModel):
    def __init__(self, **kwargs):
        super().__init__(self, **kwargs)
        self.model = CustomModel(**kwargs)
        #TimeSeqModel.__init__(self)
        #CustomModel.__init__(self, **kwargs)

    def forward(self, inputs):
        return self.model(inputs)