"""Configuration factory for joint RL/SSL loss and gradient mixers."""

from marlite.util.loss_func import PITLoss
from marlite.util.loss_mixer import WeightedSumLoss, EMAGradNormLoss, PCGradLoss


registered_loss_mixers = {
    "weighted_sum": WeightedSumLoss,
    "pit_loss": PITLoss,
    "ema_grad_norm": EMAGradNormLoss,
    "pcgrad": PCGradLoss,
}


class LossMixerConfig:
    """Build an independent stateful mixer for the trainer or each worker.

    YAML goes under ``trainer.loss_mixer``, for example::

        loss_mixer:
          type: ema_grad_norm
          weights: [1.0, 0.1]
          alpha: 0.9
          target_ratio: 0.2
          min_weight: 0.001
          max_weight: 1.0

    Task order is [RL, SSL]. ``weighted_sum`` and ``pcgrad`` use fixed weights;
    ``pit_loss`` applies weights AFTER its CDF transform (alpha/min_std optional).
    ``ema_grad_norm`` keeps weights[0] fixed and uses weights[1] as the fallback
    when shared gradients vanish. Its adaptive weight aims at target_ratio and
    is clamped to [min_weight, max_weight]. It does not learn weights by backprop.

    PIT defaults to mean reduction; other methods default to sum. A gradient
    mixer is still one joint optimizer update, not sequential RL/SSL updates.
    """

    def __init__(self, **kwargs):
        self.mixer_type = kwargs.pop("type", "weighted_sum")
        if self.mixer_type not in registered_loss_mixers:
            raise ValueError(f"Unsupported loss mixer type: {self.mixer_type}")
        self.mixer_kwargs = kwargs

    def get_loss_mixer(self):
        mixer = registered_loss_mixers[self.mixer_type](num_tasks=2, **self.mixer_kwargs)
        if mixer.reduction == "none":
            raise ValueError("RL/SSL training requires a scalar loss reduction")
        return mixer
