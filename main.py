import yaml
from absl import app, flags, logging

from marlite.trainer.trainer_config import TrainerConfig
from marlite.experiment_analyzer.experiment_analyzer_config import (
    ExperimentAnalyzerConfig,
)


_CONFIG = flags.DEFINE_string(
    "config", None, "Path to the YAML configuration file."
)
_OUTPUT = flags.DEFINE_string(
    "output", None, "Path to the analysis output YAML file."
)
_CHECKPOINT = flags.DEFINE_string(
    "checkpoint", "best", "Checkpoint name used for analysis."
)

flags.mark_flag_as_required("config")
flags.FLAGS.set_default("verbosity", logging.INFO)
flags.FLAGS.set_default("stderrthreshold", "info")


def train(config_path):
    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    TrainerConfig(config).run()
    print("Training completed.")


def analyze(config_path, output_path, checkpoint="best"):
    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    analyzer = ExperimentAnalyzerConfig(config).create_analyzer(
        checkpoint=checkpoint
    )
    results = analyzer.comprehensive_analysis()

    with open(output_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(results, file, default_flow_style=False)


def main(argv):
    if len(argv) != 2 or argv[1] not in ("train", "analyze"):
        raise app.UsageError("Expected command: train or analyze")

    command = argv[1]
    if command == "train":
        train(_CONFIG.value)
        return

    if _OUTPUT.value is None:
        raise app.UsageError("--output is required for analyze")
    analyze(_CONFIG.value, _OUTPUT.value, _CHECKPOINT.value)


if __name__ == "__main__":
    app.run(main)
