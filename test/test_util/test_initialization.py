from copy import deepcopy

import pytest
import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv

from marlite.algorithm.model.model_config import ModelConfig
from marlite.algorithm.graph_builder.g2anet_graph_builder import G2ANetGraphBuilder
from marlite.util.initialization import initialize_model


@pytest.mark.parametrize("scheme,options", [
    ("orthogonal", {"gain": .01}), ("xavier_uniform", {"gain": 1}),
    ("xavier_normal", {}), ("kaiming_uniform", {"nonlinearity": "relu"}),
    ("kaiming_normal", {"a": .1, "nonlinearity": "leaky_relu", "mode": "fan_out"}),
    ("normal", {"mean": 0, "std": .01}), ("uniform", {"a": -.1, "b": .1}),
    ("trunc_normal", {"std": .01, "a": -.02, "b": .02}), ("constant", {"value": .2}),
])
def test_methods_match_pytorch(scheme, options):
    layer = nn.Linear(8, 5)
    expected = torch.empty_like(layer.weight)
    kwargs = dict(options)
    if scheme == "constant":
        kwargs["val"] = kwargs.pop("value")
    torch.manual_seed(12)
    getattr(nn.init, scheme + "_")(expected, **kwargs)
    torch.manual_seed(12)
    initialize_model(layer, {"scheme": scheme, "bias": 0, **options})
    torch.testing.assert_close(layer.weight, expected)
    assert torch.count_nonzero(layer.bias) == 0


def test_output_slices_and_small_policy_head():
    model = nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 6))
    original = model[0].weight.clone()
    initialize_model(model, {"overrides": [{
        "module": "2", "scheme": "orthogonal", "gain": .01, "bias": 0,
    }]})
    torch.testing.assert_close(model[0].weight, original)
    torch.testing.assert_close(model[2].weight @ model[2].weight.T, torch.eye(6) * .0001)
    initialize_model(model, {"overrides": [{
        "module": "2", "scheme": "normal", "std": .01, "bias": 0,
        "output_slices": [{"start": 3, "stop": 6, "scheme": "constant", "value": 0, "bias": -1}],
    }]})
    assert model[2].weight[:3].abs().sum() > 0
    assert torch.count_nonzero(model[2].weight[3:]) == 0
    assert (model[2].bias[3:] == -1).all()


@pytest.mark.parametrize("recurrent", [nn.GRU, nn.LSTM])
def test_gatewise_recurrent_initialization(recurrent):
    layer = recurrent(5, 4, num_layers=2, bidirectional=True)
    rule = {"module": "", "scheme": "xavier_uniform", "bias": 0,
            "recurrent": {"scheme": "orthogonal"}}
    if recurrent is nn.LSTM:
        rule["forget_bias"] = 1
    initialize_model(layer, {"overrides": [rule]})
    gates = 4 if recurrent is nn.LSTM else 3
    for name, parameter in layer.named_parameters():
        if name.startswith("weight_hh"):
            for gate in parameter.chunk(gates):
                torch.testing.assert_close(gate @ gate.T, torch.eye(4), atol=1e-6, rtol=1e-6)
        if recurrent is nn.LSTM and name.startswith("bias_ih"):
            total = parameter + getattr(layer, name.replace("bias_ih", "bias_hh"))
            torch.testing.assert_close(total.chunk(4)[1], torch.ones(4))


@pytest.mark.parametrize("separate", [False, True])
def test_attention_projections_and_no_double_initialization(separate):
    layer = nn.MultiheadAttention(8, 2, kdim=4 if separate else 8, vdim=4 if separate else 8)
    expected = deepcopy(layer)
    torch.manual_seed(31)
    weights = (expected.in_proj_weight.chunk(3) if not separate
               else [expected.q_proj_weight, expected.k_proj_weight, expected.v_proj_weight])
    for weight in weights:
        nn.init.xavier_uniform_(weight)
    nn.init.xavier_uniform_(expected.out_proj.weight)
    torch.manual_seed(31)
    initialize_model(layer, {"scheme": "xavier_uniform", "bias": 0})
    for name, value in layer.named_parameters():
        if "weight" in name:
            torch.testing.assert_close(value, dict(expected.named_parameters())[name])


@pytest.mark.parametrize("layer", [nn.Conv1d(2, 3, 3), nn.Conv2d(2, 3, 3),
                                    nn.Conv3d(2, 3, 3), nn.ConvTranspose2d(2, 3, 3),
                                    GCNConv(4, 3), GATConv(4, 3, heads=2)])
def test_projection_adapters(layer):
    attention = layer.att_src.clone() if isinstance(layer, GATConv) else None
    initialize_model(layer, {"scheme": "constant", "value": .2, "bias": 0})
    for name, parameter in layer.named_parameters():
        if name.endswith("weight"):
            assert torch.all(parameter == .2)
        elif name.endswith("bias"):
            assert torch.count_nonzero(parameter) == 0
    if attention is not None:
        torch.testing.assert_close(layer.att_src, attention)
        initialize_model(layer, {"overrides": [{
            "module": "", "parameter": "att_src", "scheme": "normal", "std": .01,
        }]})
        assert not torch.equal(layer.att_src, attention)


def test_native_norm_and_explicit_parameter_override():
    model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4), nn.BatchNorm1d(4))
    initialize_model(model, {"scheme": "normal", "std": .01})
    assert (model[1].weight == 1).all()
    assert (model[2].running_var == 1).all()
    initialize_model(model, {"overrides": [{
        "module": "1", "parameter": "weight", "scheme": "constant", "value": .5,
    }]})
    assert (model[1].weight == .5).all()


def test_model_config_checkpoint_and_child_protection(tmp_path):
    cfg = dict(model_type="Custom", layers=[dict(type="Linear", in_features=4, out_features=2)])
    source = ModelConfig(**cfg).get_model()
    path = tmp_path / "params.pt"
    torch.save(source.state_dict(), path)
    child = ModelConfig(**cfg, pretrained_params_path=path,
                        initialization={"scheme": "constant", "value": 9}).get_model()
    parent = nn.Sequential(child, nn.Linear(2, 1))
    initialize_model(parent, {"scheme": "constant", "value": 7})
    for key, parameter in source.state_dict().items():
        torch.testing.assert_close(child.state_dict()[key], parameter)
    assert (parent[1].weight == 7).all()
    with pytest.raises(ValueError, match="protected"):
        initialize_model(parent, {"overrides": [{"module": "0.model.0", "scheme": "normal"}]})


def test_shared_parameters_initialized_once():
    layer = nn.Linear(4, 4)
    model = nn.ModuleList([layer, layer])
    torch.manual_seed(22)
    expected = torch.randn_like(layer.weight)
    torch.manual_seed(22)
    initialize_model(model, {"scheme": "normal"})
    torch.testing.assert_close(layer.weight, expected)
    with pytest.raises(ValueError, match="shared"):
        initialize_model(model, {"overrides": [
            {"module": "0", "scheme": "normal"}, {"module": "1", "scheme": "normal"},
        ]})


def test_g2anet_initialization_entry():
    builder = G2ANetGraphBuilder(3, input_dim=4, hidden_dim=4, initialization={"overrides": [
        {"module": "hard_attention", "scheme": "xavier_uniform", "bias": 0,
         "recurrent": {"scheme": "orthogonal"}, "forget_bias": 1},
    ]})
    assert (builder.attention_model.hard_attention.bias_ih_l0[4:8] == 1).all()


@pytest.mark.parametrize("config", [
    {"scheme": "typo"}, {"scheme": "normal", "gain": 1},
    {"scheme": "normal", "std": -1}, {"scheme": "constant"},
    {"scheme": "uniform", "a": 2, "b": 1}, {"scheme": "normal", "std": float("nan")},
    {"overrides": [{"module": "missing", "scheme": "normal"}]},
    {"overrides": [{"module": "", "scheme": "normal", "forget_bias": 1}]},
    {"overrides": [{"module": "", "parameter": "bias", "scheme": "orthogonal"}]},
    {"overrides": [{"module": "", "scheme": "normal", "output_slices": [
        {"start": 0, "stop": 10, "scheme": "normal"}]}]},
])
def test_invalid_configuration_fails(config):
    with pytest.raises(ValueError):
        initialize_model(nn.Linear(4, 3), config)


def test_unconfigured_model_keeps_native_weights_and_rng():
    torch.manual_seed(11)
    expected = nn.Linear(4, 3)
    state = torch.get_rng_state()
    torch.manual_seed(11)
    actual = ModelConfig(model_type="Custom", layers=[dict(type="Linear", in_features=4, out_features=3)]).get_model()
    torch.testing.assert_close(actual.model[0].weight, expected.weight)
    assert torch.equal(torch.get_rng_state(), state)
