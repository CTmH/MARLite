"""Sequence QPLEX reuses the factorization and respects historical masks."""

import pytest
import torch

from marlite.algorithm.critic import CriticConfig, QPLEXMixer, SeqQPLEXMixer
from test.test_algorithm.test_critic.test_qplex_mixer import _make_qplex_mixer_cfg, _make_inputs


@pytest.mark.parametrize("sequence", [
    {"model_type": "SimpleResAttSeqEnc", "input_dim": 16, "embed_dim": 16,
     "output_dim": 16, "num_heads": 2, "max_seq_len": 7, "dropout": 0.},
    {"model_type": "RNN", "input_shape": 16, "output_shape": 16, "rnn_hidden_dim": 16},
    {"model_type": "CustomConv1D", "layers": [
        {"type": "Conv1d", "in_channels": 16, "out_channels": 16, "kernel_size": 1},
        {"type": "AdaptiveAvgPool1d", "output_size": 1}, {"type": "Flatten"}]},
    {"model_type": "Identity"},
])
def test_sequence_backends_and_gradients(sequence):
    config = _make_qplex_mixer_cfg()
    config.update(type="SeqQPLEXMixer", seq_model=sequence)
    mixer = CriticConfig(**config).get_critic()
    assert isinstance(mixer, SeqQPLEXMixer) and isinstance(mixer, QPLEXMixer)
    inputs = _make_inputs()
    inputs["padding_mask"].fill_(False)
    inputs["padding_mask"][:, :2] = True
    inputs["alive_mask"][1, -1] = False
    inputs["states"].requires_grad_(True)
    out = mixer(**inputs)
    assert torch.isfinite(out["q_tot"]).all()
    assert out["q_tot"][1] == 0
    out["q_tot"].sum().backward()
    grad = inputs["states"].grad
    assert torch.isfinite(grad).all() and grad[:, 2:].abs().sum() > 0
    assert grad[:, :2].count_nonzero() == 0
    if sequence["model_type"] != "Identity":
        assert grad[0, 2:-1].abs().sum() > 0  # History, not only the latest state.
    mixer.load_state_dict(mixer.state_dict())


def test_attention_masks_dead_rows_and_empty_histories():
    config = _make_qplex_mixer_cfg()
    config.update(type="SeqQPLEXMixer", feature_extractor={
        "model_type": "SimpleResAttMaskedStateEnc", "input_dim": 5,
        "embed_dim": 16, "num_heads": 2, "max_seq_len": 3, "dropout": 0.,
    }, seq_model={
        "model_type": "SimpleResAttSeqEnc", "input_dim": 16, "embed_dim": 16,
        "output_dim": 16, "num_heads": 2, "max_seq_len": 7, "dropout": 0.,
    })
    mixer = CriticConfig(**config).get_critic().eval()
    inputs = _make_inputs()
    inputs["states"] = torch.randn(4, 7, 3, 5, requires_grad=True)
    inputs["padding_mask"].fill_(False)
    inputs["padding_mask"][0, :2] = True
    inputs["padding_mask"][2] = True
    inputs["alive_mask"][0, 2, 1] = False
    inputs["alive_mask"][3] = False
    before = mixer(**inputs)["q_tot"]
    assert torch.isfinite(before).all() and before[3] == 0
    changed = inputs["states"].detach().clone()
    changed[0, :2] += 100
    changed[0, 2, 1] += 100
    changed[2:] += 100
    after = mixer(**{**inputs, "states": changed})["q_tot"]
    torch.testing.assert_close(before, after)
    before.sum().backward()
    assert torch.isfinite(inputs["states"].grad).all()
