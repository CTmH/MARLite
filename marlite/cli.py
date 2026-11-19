"""One command entry point for local and single-node torchrun experiments."""

import argparse
import json
from pathlib import Path

import torch.distributed as dist
import yaml

from marlite.configuration.io import load_config
from marlite.configuration.recipe import build_experiment
from marlite.configuration.registry import ConfigurationError


def _parser():
    parser = argparse.ArgumentParser(prog="marlite")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "validate", "analyze"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--device", default="auto", help="auto, cpu, mps, cuda or cuda:index")
        if name in ("train", "validate"):
            command.add_argument("--workdir", type=Path, required=name == "train")
        if name == "train":
            command.add_argument("--resume", type=Path)
            command.add_argument(
                "--iterations", type=int, help="Additional online training iterations"
            )
        if name == "analyze":
            command.add_argument("--checkpoint", type=Path, required=True)
            command.add_argument("--episodes", type=int, default=32)
            command.add_argument("--test-seed", type=int)
            command.add_argument("--evaluation-round", type=int, default=0)
            command.add_argument("--stochastic", action="store_true")
            command.add_argument("--output", type=Path)
            command.add_argument("--trajectories", type=Path)
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "analyze":
            evaluator = build_experiment(config, device=args.device, policy_only=True)
            evaluator.load_checkpoint(args.checkpoint)
            report = evaluator.evaluate(
                args.episodes,
                seed=args.test_seed,
                round_index=args.evaluation_round,
                stochastic=args.stochastic,
                trajectories=args.trajectories,
            )
            if evaluator.backend.rank == 0:
                output = json.dumps(report, ensure_ascii=False, indent=2)
                if args.output is None:
                    print(output)
                else:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    if args.output.suffix.lower() in (".yaml", ".yml"):
                        output = yaml.safe_dump(report, allow_unicode=True, sort_keys=False)
                    args.output.write_text(output + "\n", encoding="utf-8")
        else:
            experiment = build_experiment(config, workdir=args.workdir, device=args.device)
            if args.command == "validate":
                if experiment.trainer.learner.backend.rank == 0:
                    print(
                        json.dumps(
                            {
                                "recipe": config["recipe"],
                                "environment": experiment.environment_spec.state_dict(),
                                "actor": str(experiment.trainer.learner.actor),
                            },
                            ensure_ascii=False,
                            indent=2,
                        )
                    )
            else:
                run_options = dict(config.get("run", {}))
                if args.iterations is not None:
                    run_options["iterations"] = args.iterations
                experiment.run(resume=args.resume, run_options=run_options)
    except (ConfigurationError, FileExistsError) as error:
        parser.exit(2, str(error) + "\n")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
