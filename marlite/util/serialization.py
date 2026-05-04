import io
import torch


def get_state_dict(model):
    if hasattr(model, '_orig_mod'):
        return model._orig_mod.state_dict()
    return model.state_dict()


def load_state_dict_into(model, state_dict, strict=True):
    if hasattr(model, '_orig_mod'):
        return model._orig_mod.load_state_dict(state_dict, strict=strict)
    return model.load_state_dict(state_dict, strict=strict)


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
