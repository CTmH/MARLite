"""Actual package CLI and torchrun workflows for version-3 configurations."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist
import yaml

from marlite.configuration.recipe import build_experiment
from test.test_configuration import minimal_config
from test.sequence_test_tools import run_torchrun


def mpe_config(recipe):
    config = minimal_config("mpe2.simple_spread_v3", recipe)
    config["environment"]["params"] = {"N": 2, "max_cycles": 3, "continuous_actions": False}
    config["collector"]["max_steps"] = 3
    return config


def test_package_cli_train_resume_and_actor_only_analyze(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(mpe_config("qmpd")), encoding="utf-8")
    workdir = tmp_path / "run"
    root = Path(__file__).parents[1]
    env = dict(
        os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1"
    )

    def command(*args):
        result = subprocess.run(
            [sys.executable, "-m", "marlite", *args],
            env=env,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=45,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return result

    command("validate", "--config", str(config), "--device", "cpu")
    command("train", "--config", str(config), "--workdir", str(workdir), "--device", "cpu")
    checkpoint = workdir / "checkpoints" / "final.pt"
    command(
        "train",
        "--config",
        str(config),
        "--workdir",
        str(workdir),
        "--device",
        "cpu",
        "--resume",
        str(checkpoint),
        "--iterations",
        "1",
    )
    state = torch.load(checkpoint, weights_only=True)
    assert state["counters"]["iterations"] == 2
    output, trajectories = tmp_path / "report.json", tmp_path / "episodes.pt"
    command(
        "analyze",
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--device",
        "cpu",
        "--episodes",
        "2",
        "--test-seed",
        "997",
        "--output",
        str(output),
        "--trajectories",
        str(trajectories),
    )
    report = json.loads(output.read_text())
    assert report["episode_length"]["count"] == 2 and report["evaluation_seed"] == 997
    assert len(torch.load(trajectories, weights_only=True)["episodes"]) == 2


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo unavailable")
def test_recipe_torchrun_training_matches_single_process(tmp_path):
    expected = {}
    for recipe in ("mappo", "qmix", "qplex", "qtran", "qmpd"):
        experiment = build_experiment(mpe_config(recipe), workdir=tmp_path / ("single_" + recipe))
        experiment.run()
        expected[recipe] = experiment.trainer.learner.actor.state_dict()
        (tmp_path / (recipe + ".yaml")).write_text(
            yaml.safe_dump(mpe_config(recipe)), encoding="utf-8"
        )
    run_torchrun(
        tmp_path,
        """
import os
from pathlib import Path
import torch
import torch.distributed as dist
from marlite.configuration.io import load_config
from marlite.configuration.recipe import build_experiment
from marlite.training.distributed import initialize_backend

backend = initialize_backend("cpu", timeout_seconds=60)
root = Path(os.environ["TEST_RUN_DIR"])
results = {}
try:
    for recipe in ("mappo", "qmix", "qplex", "qtran", "qmpd"):
        config = load_config(root / (recipe + ".yaml"))
        experiment = build_experiment(config, backend=backend, workdir=root / recipe)
        experiment.run()
        if backend.rank == 0:
            results[recipe] = experiment.trainer.learner.actor.state_dict()
    # Unseeded runs must still partition the SAME sampler order on every rank.
    config["seed"] = None
    experiment = build_experiment(config, backend=backend, workdir=root / "unseeded")
    states = backend.gather_objects(experiment.trainer.rng.get_state())
    assert all(torch.equal(states[0], state) for state in states)
    if backend.rank == 0:
        torch.save(results, root / "result.pt")
finally:
    dist.destroy_process_group()
""",
        timeout=150,
    )
    actual = torch.load(tmp_path / "result.pt", weights_only=True)
    for recipe, parameters in expected.items():
        for name, parameter in parameters.items():
            other = actual[recipe][name]
            if name.endswith("in_proj_bias"):
                # Shared key biases are a separately tested softmax-null direction.
                query, _, value = parameter.chunk(3)
                other_query, _, other_value = other.chunk(3)
                torch.testing.assert_close(query, other_query, rtol=3e-5, atol=3e-6)
                torch.testing.assert_close(value, other_value, rtol=3e-5, atol=3e-6)
            else:
                torch.testing.assert_close(parameter, other, rtol=3e-5, atol=3e-6)


def test_qmpd_cli_resumes_interrupted_offline_training(tmp_path):
    from test.test_sequence_mappo import assert_tree_equal

    settings = mpe_config("qmpd")
    settings["offline"].update(codec_updates=3, policy_updates=2, checkpoint_interval=1)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(settings), encoding="utf-8")
    root = Path(__file__).parents[1]
    device = os.environ.get("MARLITE_ACCEPTANCE_DEVICE", "cpu")

    def command(workdir, *options, interrupt=False):
        invocation = ["-m", "marlite"]
        if interrupt:
            # Stop at a known checkpoint boundary without a race with a fast machine.
            invocation = [
                "-c",
                """
from pathlib import Path
from marlite.cli import main
from marlite.training.distillation import SequenceDistillationTrainer

save = SequenceDistillationTrainer.save_checkpoint

def interrupt_after_save(self, path):
    save(self, path)
    if Path(path).name == "offline_1.pt":
        raise SystemExit(73)

SequenceDistillationTrainer.save_checkpoint = interrupt_after_save
main()
""",
            ]
        result = subprocess.run(
            [
                sys.executable,
                *invocation,
                "train",
                "--config",
                str(config),
                "--workdir",
                str(workdir),
                "--device",
                device,
                *options,
            ],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == (73 if interrupt else 0), result.stdout + result.stderr

    interrupted = tmp_path / "interrupted"
    reference = tmp_path / "uninterrupted"
    command(interrupted, interrupt=True)
    checkpoint = interrupted / "checkpoints/offline_1.pt"
    state = torch.load(checkpoint, weights_only=True)
    assert state["counters"]["iterations"] == 0
    assert state["extra"]["offline_counters"] == {"probe": 0, "codec": 1, "policy": 0}
    log = interrupted / "offline_events.jsonl"
    # An interrupted process can log an update before its next checkpoint is saved.
    with log.open("a", encoding="utf-8") as stream:
        stream.write('{"stage":"codec","update":2,"training":{}}\n')

    command(interrupted, "--resume", str(checkpoint))
    command(reference)
    resumed, uninterrupted = [
        torch.load(run / "checkpoints/final.pt", weights_only=True)
        for run in (interrupted, reference)
    ]
    for key in ("learner", "extra", "counters", "sampler_rng"):
        assert_tree_equal(resumed[key], uninterrupted[key])
    assert log.read_text() == (reference / log.name).read_text()
    assert len((interrupted / "teacher/events.jsonl").read_text().splitlines()) == 1
