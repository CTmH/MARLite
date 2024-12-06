from .rnn import RNNModel

REGISTERED_MODELS = {
    "RNN": RNNModel,
}

class ModelConfig:
    def __init__(self, model_type: str = "RNN", layers: dict = None, num_layers: int = 6,
                 hidden_size: int = 512, num_heads: int = 8,
                 dropout_rate: float = 0.1):
        self.model_type = model_type
        self.layers = layers or {}  # Dictionary to hold specific layer configurations if needed.
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

    def __str__(self):
        return f"ModelConfig(model_type={self.model_type}, num_layers={self.num_layers}, " \
               f"hidden_size={self.hidden_size}, num_heads={self.num_heads}, dropout_rate={self.dropout_rate})"
    
    def get_model(self):
        model = REGISTERED_MODELS.get(self.model_type)
        if model is None:
            raise ValueError(f"Model type {self.model_type} not registered.")
        return model(self.layers)
