"""Save resolved experiment settings and enough code/version evidence to reproduce them."""

import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import zipfile

import torch
import yaml


def save_experiment_metadata(experiment):
    trainer = experiment.trainer
    backend, workdir = trainer.learner.backend, trainer.workdir
    if workdir is None:
        return
    error = None
    if backend.rank == 0:
        try:
            workdir.mkdir(parents=True, exist_ok=True)
            root = Path(__file__).resolve().parents[2]
            source = sorted((root / "marlite").rglob("*.py"))
            source += [
                p
                for name in ("pyproject.toml", "uv.lock", "main.py")
                if (p := root / name).is_file()
            ]
            files = {
                str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in source
            }
            digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
            archive = workdir / "source" / (digest[:16] + ".zip")
            archive.parent.mkdir(exist_ok=True)
            if not archive.exists():
                with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
                    for path in source:
                        output.write(path, path.relative_to(root))
            git = {}
            for key, arguments in (
                ("commit", ["rev-parse", "HEAD"]),
                ("branch", ["branch", "--show-current"]),
                ("status", ["status", "--porcelain"]),
            ):
                result = subprocess.run(
                    ["git", *arguments], cwd=root, capture_output=True, text=True, timeout=5
                )
                git[key] = result.stdout.strip() if result.returncode == 0 else None
            versions = {"python": platform.python_version(), "torch": str(torch.__version__)}
            for dependency in (
                "mpe2",
                "magent2",
                "smac-pettingzoo",
                "sumo-rl",
                "pettingzoo",
                "numpy",
                "pyyaml",
            ):
                try:
                    versions[dependency] = metadata.version(dependency)
                except metadata.PackageNotFoundError:
                    versions[dependency] = None
            record = {
                "versions": versions,
                "git": git,
                "source_digest": digest,
                "source_archive": str(archive.relative_to(workdir)),
                "files": files,
                "device": str(backend.device),
                "world_size": backend.world_size,
                "environment": experiment.environment_spec.state_dict(),
            }
            (workdir / "resolved.yaml").write_text(
                yaml.safe_dump(experiment.config, sort_keys=False), encoding="utf-8"
            )
            (workdir / "provenance.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            error = repr(exc)
    error = backend.broadcast_object(error)
    if error is not None:
        raise RuntimeError("Saving experiment metadata failed: " + error)
