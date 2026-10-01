"""Analytic regression checks for optimization direction and gradient mixing."""

from copy import deepcopy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from marlite.util.loss_func import PITLoss
from marlite.util.loss_mixer import EMAGradNormLoss, PCGradLoss, WeightedSumLoss, reduce_mixed_gradients
from marlite.util.loss_mixer_config import LossMixerConfig
from marlite.config_processor.qmix_config_processor import SemiSupervisedQMIXConfigProcessor


@pytest.mark.parametrize("reduction", ["sum", "mean", "none"])
def test_pit_monotone_initial_improving_and_negative_losses(reduction):
    mixer = PITLoss(2, reduction=reduction)
    for values in ([1., 1.], [.9, 1.1], [-.1, .2], [-.2, .1]):
        losses = torch.tensor(values, requires_grad=True)
        mixer(losses).sum().backward()
        assert torch.isfinite(losses.grad).all()
        assert (losses.grad > 0).all()
    mixer.eval()
    state = deepcopy(mixer.state_dict())
    mixer(torch.tensor([-.1, .1]))
    for key, value in state.items():
        torch.testing.assert_close(mixer.state_dict()[key], value)
    # Evaluation before any training must also be finite.
    assert torch.isfinite(PITLoss(2).eval()(torch.tensor([1., 1.])))


def test_weighted_sum_signed_losses_and_pit_post_transform_weights():
    losses = torch.tensor([-2., 3.], requires_grad=True)
    mixed = WeightedSumLoss(weights=[1., .2])(losses)
    mixed.backward()
    torch.testing.assert_close(mixed, torch.tensor(-1.4))
    torch.testing.assert_close(losses.grad, torch.tensor([1., .2]))
    for weight in (.1, 1.):
        losses = torch.tensor([1., 1.], requires_grad=True)
        PITLoss(2, weights=[1., weight])(losses).backward()
        torch.testing.assert_close(losses.grad[1] / losses.grad[0], torch.tensor(weight))


def test_ema_balances_shared_gradients_and_preserves_private_heads():
    shared, rl_head, ssl_head = [torch.nn.Parameter(torch.tensor(1.)) for _ in range(3)]
    mixer = EMAGradNormLoss(weights=[1., .25], target_ratio=.5)
    mixer((2 * shared + 100 * rl_head, 10 * shared + ssl_head),
          parameters=[shared, rl_head, ssl_head]).backward()
    torch.testing.assert_close(mixer.effective_ssl_weight, torch.tensor(.1))
    torch.testing.assert_close(shared.grad, torch.tensor(3.))
    torch.testing.assert_close(rl_head.grad, torch.tensor(100.))
    torch.testing.assert_close(ssl_head.grad, torch.tensor(.1))
    state = deepcopy(mixer.state_dict())
    # A closed RL gate must not drive the SSL weight towards zero.
    mixer((0 * shared, 10 * shared), parameters=[shared])
    torch.testing.assert_close(mixer.effective_ssl_weight, torch.tensor(.25))
    torch.testing.assert_close(mixer.norm_ema, state["norm_ema"])
    restored = EMAGradNormLoss(weights=[1., .25], target_ratio=.5)
    restored.load_state_dict(state)
    assert restored.initialized
    torch.testing.assert_close(restored.norm_ema, torch.tensor([2., 10.]))


@pytest.mark.parametrize("second,expected", [([-1., 1.], [.5, 1.5]),
                                           ([1., 1.], [2., 1.]),
                                           ([-1., 0.], [0., 0.])])
def test_pcgrad_projection_private_unused_and_zero_gradients(second, expected):
    shared = torch.nn.Parameter(torch.tensor([1., 2.]))
    private = torch.nn.Parameter(torch.tensor(3.))
    unused = torch.nn.Parameter(torch.tensor(0.))
    losses = (shared[0] + 2 * private, shared @ torch.tensor(second))
    mixed = PCGradLoss()(losses, parameters=[shared, private, unused])
    torch.testing.assert_close(mixed, sum(losses))
    mixed.backward()
    torch.testing.assert_close(shared.grad, torch.tensor(expected))
    torch.testing.assert_close(private.grad, torch.tensor(2.))
    assert unused.grad is None


@pytest.mark.parametrize("kind", ["pit_loss", "weighted_sum", "ema_grad_norm", "pcgrad"])
def test_config_factory_validation_and_zero_weight(kind):
    config = {"loss_mixer": {"type": kind, "weights": [1., 0.]}}
    SemiSupervisedQMIXConfigProcessor.parse_loss_mixer_config(config)
    assert isinstance(config["loss_mixer_config"], LossMixerConfig)
    mixer = config["loss_mixer_config"].get_loss_mixer()
    parameter = torch.nn.Parameter(torch.tensor(1.))
    mixer((parameter, -100 * parameter), parameters=[parameter]).backward()
    assert parameter.grad > 0
    with pytest.raises(ValueError):
        mixer(torch.tensor([float("nan"), 1.]), parameters=[parameter])
    with pytest.raises(ValueError):
        mixer(torch.ones(3), parameters=[parameter])


def test_config_rejects_conflicting_options():
    with pytest.raises(ValueError):
        SemiSupervisedQMIXConfigProcessor.parse_loss_mixer_config(
            {"loss_mixer": {"type": "pcgrad"}, "loss_combination_method": "pit_loss"})
    with pytest.raises(ValueError):
        LossMixerConfig(type="typo")
    legacy = {"loss_combination_method": "pit_loss", "pit_loss_alpha": .8,
              "self_supervised_learning_loss_weight": .25}
    with pytest.warns(FutureWarning):
        SemiSupervisedQMIXConfigProcessor.parse_loss_mixer_config(legacy)
    mixer = legacy["loss_mixer_config"].get_loss_mixer()
    assert mixer.alpha == .8
    torch.testing.assert_close(mixer.weights, torch.tensor([1., .25]))


@pytest.mark.parametrize("mixer_class", [EMAGradNormLoss, PCGradLoss])
def test_gradient_mixer_disjoint_parameters_and_tensor_interface(mixer_class):
    first, second = [torch.nn.Parameter(torch.tensor(1.)) for _ in range(2)]
    mixer = mixer_class(weights=[1., .2])
    mixer((2 * first, 10 * second), parameters=[first, second]).backward()
    torch.testing.assert_close(first.grad, torch.tensor(2.))
    torch.testing.assert_close(second.grad, torch.tensor(2.))
    shared = torch.nn.Parameter(torch.tensor(1.))
    mixer(torch.stack([2 * shared, 10 * shared]), parameters=[shared]).backward()
    assert torch.isfinite(shared.grad)
    state = deepcopy(mixer.state_dict())
    mixer.eval()
    mixer((shared, shared), parameters=[shared])
    for key, value in state.items():
        torch.testing.assert_close(mixer.state_dict()[key], value)


@pytest.mark.parametrize("kind", ["pit_loss", "weighted_sum", "ema_grad_norm", "pcgrad"])
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_forward_backward(kind):
    parameter = torch.nn.Parameter(torch.tensor([1., 2.], device="mps"))
    mixer = LossMixerConfig(type=kind).get_loss_mixer()
    mixer((parameter[0], parameter[1] - parameter[0]), parameters=[parameter]).backward()
    assert parameter.grad.device.type == "mps"
    assert torch.isfinite(parameter.grad).all()


def _distributed_check(rank, rendezvous, backend="gloo"):
    if backend == "nccl":
        torch.cuda.set_device(rank)
        torch.set_default_device(f"cuda:{rank}")
    dist.init_process_group(backend, init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        rl_vectors = [torch.tensor([2., -1.]), torch.tensor([0., 3.])]
        ssl_vectors = [torch.tensor([-3., 2.]), torch.tensor([-1., 0.])]
        for kind in ("weighted_sum", "pit_loss", "ema_grad_norm", "pcgrad"):
            config = LossMixerConfig(type=kind, weights=[1., .5])
            mixer, reference = config.get_loss_mixer(), config.get_loss_mixer()
            for _ in range(3):
                p = torch.nn.Parameter(torch.tensor([.2, .3]))
                expected_p = torch.nn.Parameter(p.detach().clone())
                private = torch.nn.Parameter(torch.tensor(.4))
                expected_private = torch.nn.Parameter(private.detach().clone())
                unused = torch.nn.Parameter(torch.tensor(1.))
                losses = (p @ rl_vectors[rank] + (private if rank == 0 else 0),
                          p @ ssl_vectors[rank])
                global_losses = (expected_p @ ((rl_vectors[0] + rl_vectors[1]) / 2)
                                 + .5 * expected_private,
                                 expected_p @ ((ssl_vectors[0] + ssl_vectors[1]) / 2))
                mixer(losses, parameters=[p, private, unused], distributed=True).backward()
                reduce_mixed_gradients([p, private, unused])
                reference(global_losses, parameters=[expected_p, expected_private]).backward()
                torch.testing.assert_close(p.grad, expected_p.grad)
                torch.testing.assert_close(private.grad, expected_private.grad)
                assert unused.grad is None
                for key, value in mixer.state_dict().items():
                    torch.testing.assert_close(value, reference.state_dict()[key])
    finally:
        dist.destroy_process_group()
        torch.set_default_device("cpu")


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo unavailable")
def test_two_process_mixing_matches_global_objective(tmp_path):
    mp.spawn(_distributed_check, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two GPUs required")
def test_two_gpu_mixing_matches_global_objective(tmp_path):
    mp.spawn(_distributed_check,
             args=(f"file://{tmp_path / 'gpu-rendezvous'}", "nccl"), nprocs=2)
