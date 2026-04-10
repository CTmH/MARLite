import io
import torch


def serialize_to_buffer(state_dict):
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    buffer.seek(0)
    return buffer.getvalue()


def deserialize_from_buffer(buffer):
    if isinstance(buffer, bytes):
        buffer = io.BytesIO(buffer)
    buffer.seek(0)
    return torch.load(buffer, weights_only=True)
