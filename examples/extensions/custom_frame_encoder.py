"""Run MAPPO with a custom local observation encoder selected in YAML."""

import argparse
from pathlib import Path

from torch import Tensor, nn

from marlite.configuration.catalog import default_registry
from marlite.configuration.io import load_config
from marlite.configuration.recipe import build_experiment


class NormalizedFrameEncoder(nn.Module):
    """Project one observation at a time and normalize its feature vector."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, observations: Tensor) -> Tensor:
        """Encode a batch of observations into one feature vector per row."""
        return self.projection(observations)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()

    registry = default_registry()
    registry.register("NormalizedFrameEncoder", NormalizedFrameEncoder)
    config = load_config(Path(__file__).with_suffix(".yaml"))
    experiment = build_experiment(config, device="cpu", workdir=args.workdir, registry=registry)
    experiment.run()


if __name__ == "__main__":
    main()
