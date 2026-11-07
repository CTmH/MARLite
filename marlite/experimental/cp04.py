"""Experimental CP04 workflow. Execute with python -m marlite.experimental.cp04."""

import argparse
import math
from copy import deepcopy
from hashlib import file_digest
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from time import perf_counter

from gymnasium.spaces import Discrete
import numpy as np
import torch
import yaml

from marlite.algorithm.agents import AgentGroupConfig
from marlite.algorithm.model.cp04_baselines import CP04_BASELINES
from marlite.environment import EnvConfig
from marlite.experimental.cp04_budget import AuditedRollouts
from marlite.rollout.persistent_env_rollout import persistent_env_rollout
from marlite.trainer.trainer_config import TrainerConfig
from marlite.util.communication_curriculum import CommunicationCurriculum, action_distillation
from marlite.util.complementary_targets import ComplementaryAttackQueries
from marlite.util.randomness import EVALUATION_STREAM, ROLLOUT_STREAM, derive_seed
from marlite.util.serialization import (
    get_state_dict,
    load_state_dict_into,
    serialize_to_buffer,
)
from marlite.util.trajectory_dataset import (
    TrajectoryDataLoader,
    TrajectoryDataset,
    trajectory_collate_fn,
)

_ALGORITHMS = {
    "MAPPO": ("CP03MAPPO", "GroupCommunicationMAPPO"),
    "QPLEX": ("CP03QPLEX", "GroupCommunicationQMIX"),
}
_WINDOW_ATTRS = ("observations", "states", "alive_mask", "group_indices", "avail_actions")
_SPLITS = ("offline_train", "offline_validation", "online_validation", "test")
_DIAGNOSTIC_SEEDS = set(range(840000, 840032)) | set(range(850000, 850032)) | set(range(860000, 860008))
_GUIDANCE_FIELDS = ("guidance_weight", "guidance_decay_updates", "guidance_temperature")


def _read_config(path):
    with Path(path).open() as stream:
        return yaml.safe_load(stream)


def _digest(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def _load_weights(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def _agent_spec(config):
    return {
        key: deepcopy(value)
        for key, value in config["agent_group"].items()
        if key not in ("optimizer", "lr_scheduler")
    }


def prepare(source_config, source_actor, output, seed):
    """Build one strictly loaded teacher and one shared student/critic initial state."""
    source = _read_config(source_config)
    config = deepcopy(source)
    algorithm = config["trainer"]["type"]
    if algorithm not in _ALGORITHMS:
        raise ValueError("CP04 requires a MAPPO or QPLEX source")
    legacy, stock = _ALGORITHMS[algorithm]
    raw = config["agent_group"]
    if (
        raw["type"] not in (legacy, stock)
        or raw.get("message_mode") != "raw"
        or raw.get("teacher_checkpoint") is not None
        or raw.get("distillation_steps", 0)
    ):
        raise ValueError("Expected a matching raw CP03 actor without a nested teacher")
    raw["type"] = stock
    for key in _GUIDANCE_FIELDS:
        raw.pop(key, None)

    teacher_spec = _agent_spec(config)
    teacher = AgentGroupConfig(**teacher_spec).get_agent_group()
    teacher.load_state_dict(_load_weights(source_actor), strict=True)
    teacher.requires_grad_(False).eval()

    student = config["agent_group"]
    student.update(message_mode="quantized", encoder_lr_scale=0.25)
    if (student.get("message_dim", 32), student.get("bits", 4)) != (32, 4):
        raise ValueError("CP04 expects 32 four-bit codes per sender")
    config["trainer"].pop("checkpoint", None)
    config["trainer"]["seed"] = seed

    # The raw checkpoint's unused codec must not become the student initialization.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        actor = AgentGroupConfig(**_agent_spec(config)).get_agent_group()
        initial = dict(teacher.state_dict())
        initial.update(
            (name, value)
            for name, value in actor.state_dict().items()
            if name.startswith("codec.") or name == "distillation_updates"
        )
        actor.load_state_dict(initial, strict=True)
        critic = TrainerConfig(config).trainer_kwargs["critic_config"].get_critic()

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    artifacts = {
        "teacher.pth": teacher.state_dict(),
        "actor_init.pth": actor.state_dict(),
        "critic_init.pth": critic.state_dict(),
    }
    for name, state in artifacts.items():
        torch.save(state, output / name)
    manifest = {
        "experimental": True,
        "algorithm": algorithm,
        "seed": seed,
        "source_config_sha256": _digest(source_config),
        "source_actor_sha256": _digest(source_actor),
        "artifacts": {name: _digest(output / name) for name in artifacts},
        "teacher": teacher_spec,
        "student_config": config,
        "teacher_behavior_verified": False,
    }
    with (output / "manifest.yaml").open("w") as stream:
        yaml.safe_dump(manifest, stream, sort_keys=False)
    return manifest


def load_bundle(directory):
    """Check the prepared artifacts before using them in another stage."""
    directory = Path(directory)
    manifest = _read_config(directory / "manifest.yaml")
    if set(manifest["artifacts"]) != {"teacher.pth", "actor_init.pth", "critic_init.pth"}:
        raise ValueError("Unexpected CP04 artifact names")
    for name, expected in manifest["artifacts"].items():
        if _digest(directory / name) != expected:
            raise ValueError(f"CP04 artifact hash mismatch: {name}")
    return manifest


def _collect_teacher(config, teacher_spec, teacher_path, seeds):
    return _rollout_actor(config, teacher_spec, _load_weights(teacher_path), seeds)


def _rollout_actor(config, agent_spec, state, seeds, *, greedy=False):
    rollout = config["rollout"]
    params = serialize_to_buffer(state)
    memory = SharedMemory(create=True, size=len(params))
    try:
        memory.buf[:len(params)] = params
        return persistent_env_rollout(
            EnvConfig(**config["environment"]),
            AgentGroupConfig(**agent_spec),
            (memory.name, len(params)),
            n_episodes=len(seeds),
            rnn_traj_len=rollout["traj_len"],
            episode_limit=rollout["episode_limit"],
            epsilon=0.0,
            required_attrs=rollout["required_attrs"],
            episode_seeds=seeds,
            greedy_evaluation=greedy,
            deterministic=config["trainer"].get("deterministic", False),
        )
    finally:
        memory.close()
        memory.unlink()


def _pack_episode(episode, senders):
    """Store one tensor per field, rather than repeating overlapping histories."""
    packed = {}
    length = episode["episode_length"]
    if length <= 0 or any(len(episode[key]) != length for key in _WINDOW_ATTRS):
        raise ValueError("Teacher episode has incomplete window fields")
    for key in _WINDOW_ATTRS:
        values = episode[key]
        if key == "avail_actions" and all(
            isinstance(step[sender], Discrete) for step in values for sender in senders
        ):
            continue
        if key == "states":
            array = np.stack(values)
        else:
            array = np.stack([np.stack([step[sender] for sender in senders]) for step in values])
        if key == "avail_actions" and (
            array.ndim != 3 or array.dtype.kind not in "biuf"
            or not np.isin(array, (0, 1)).all()
        ):
            raise ValueError("Offline action availability must be discrete spaces or masks")
        packed[key] = torch.from_numpy(array)
    return packed


def collect(bundle, split_path, output):
    """Collect only the two offline splits using the frozen raw teacher."""
    bundle = Path(bundle)
    manifest = load_bundle(bundle)
    splits = _read_config(split_path)
    if not isinstance(splits, dict) or set(splits) != set(_SPLITS):
        raise ValueError(f"Provide the four CP04 seed lists: {_SPLITS}")
    if any(
        not isinstance(seeds, list) or not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        for seeds in splits.values()
    ):
        raise ValueError("CP04 splits must be nonempty lists of nonnegative integer seeds")
    seeds = [seed for key in _SPLITS for seed in splits[key]]
    if len(seeds) != len(set(seeds)) or _DIAGNOSTIC_SEEDS.intersection(seeds):
        raise ValueError("CP04 splits overlap one another or the CP03 diagnostic maps")

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = manifest["student_config"]
    teacher = manifest["teacher"]
    senders = list(teacher["agent_list"])
    files = {}
    steps = {}
    started = perf_counter()
    for split in _SPLITS[:2]:
        episodes = _collect_teacher(config, teacher, bundle / "teacher.pth", splits[split])
        if len(episodes) != len(splits[split]):
            raise RuntimeError(f"Incomplete teacher rollout for {split}")
        path = output / f"{split}.pth"
        torch.save([_pack_episode(episode, senders) for episode in episodes], path)
        files[path.name] = _digest(path)
        steps[split] = sum(episode["episode_length"] for episode in episodes)
    with (output / "manifest.yaml").open("w") as stream:
        yaml.safe_dump({
            "experimental": True,
            "teacher_sha256": manifest["artifacts"]["teacher.pth"],
            "senders": senders,
            "history": config["rollout"]["traj_len"],
            "splits": splits,
            "artifacts": files,
            "environment_steps": steps,
            "collection_seconds": perf_counter() - started,
        }, stream, sort_keys=False)


def load_offline_dataset(directory, split, teacher_sha256):
    """Recreate MARLite's recorded windows without loading pickle-backed arrays."""
    if split not in _SPLITS[:2]:
        raise ValueError("Only offline training or validation windows can be loaded")
    directory = Path(directory)
    manifest = _read_config(directory / "manifest.yaml")
    if manifest["teacher_sha256"] != teacher_sha256:
        raise ValueError("Offline data belong to a different CP04 teacher")
    path = directory / f"{split}.pth"
    if _digest(path) != manifest["artifacts"][path.name]:
        raise ValueError(f"CP04 data hash mismatch: {split}")
    senders = manifest["senders"]
    packed = _load_weights(path)
    attrs = tuple(packed[0])
    episode_buffer = {}
    ids = []
    for episode_id, episode in enumerate(packed):
        if tuple(episode) != attrs:
            raise ValueError("Inconsistent offline window fields")
        columns = {key: value.numpy() for key, value in episode.items()}
        episode_buffer[episode_id] = {
            key: list(column) if key == "states" else [
                dict(zip(senders, row, strict=True)) for row in column
            ]
            for key, column in columns.items()
        }
        ids.extend((episode_id, step) for step in range(len(columns["states"])))
    return TrajectoryDataset(ids, episode_buffer, manifest["history"], required_attrs=attrs)


def initialize_online(trainer, actor_weights, critic_weights):
    """Load pure student weights before the first fresh online rollout."""
    actor = trainer.eval_agent_group
    if getattr(actor, "message_mode", None) != "quantized" or len(trainer.replaybuffer.buffer):
        raise ValueError("CP04 handoff requires a fresh quantized trainer")
    current = get_state_dict(actor)
    expected = {name for name in current if not name.startswith("teacher.")}
    if set(actor_weights) != expected:
        raise ValueError("CP04 handoff must contain exactly the pure student weights")
    if actor_weights["distillation_updates"].item() != 0:
        raise ValueError("Offline checkpoints cannot advance the online distillation counter")
    load_state_dict_into(actor, {**current, **actor_weights}, strict=True)
    load_state_dict_into(trainer.eval_critic, critic_weights, strict=True)
    if hasattr(trainer, "target_agent_group"):
        load_state_dict_into(trainer.target_agent_group, get_state_dict(actor), strict=True)
        load_state_dict_into(trainer.target_critic, critic_weights, strict=True)


def _offline_inputs(batch):
    obs = batch["observations"].float().transpose(1, 2)
    alive = batch["alive_mask"][:, -1].bool()
    groups = batch["group_indices"][:, -1].long()
    padding = batch["timestep_padding_mask"][:, None].bool().expand(-1, alive.shape[1], -1)
    available = batch.get("avail_actions")
    available = available[:, -1].bool() if isinstance(available, torch.Tensor) else None
    return obs, padding, alive, groups, available


def _offline_batch(dataset, batch_size, seed):
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:batch_size]
    return trajectory_collate_fn([dataset[int(index)] for index in indices])


def _query_spec(config):
    env = config["environment"]
    if env["env_name"] != "complementary_pursuit":
        raise ValueError("CP04 relation labels require the complementary-pursuit state")
    if not all(env["env_params"].get(key) for key in ("minimap_mode", "extra_features")):
        raise ValueError("CP04 group proposals need native position observation channels")
    names = ("evader_hp", "effective_damage", "ineffective_damage")
    return {
        "proposal": "group_geometry",
        **{name: env["env_params"][name] for name in names if name in env["env_params"]},
    }


def _fixed_relations(batch, alive, groups, query_spec, seed):
    valid = (
        torch.nn.functional.one_hot(groups.masked_fill(~alive, 0), alive.shape[1])
        .bool() & alive[..., None]
    ).any(dim=1)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return ComplementaryAttackQueries(**query_spec)(batch, valid, device="cpu")


def _probe_batch(teacher, batch, query_spec, seed):
    obs, padding, alive, groups, _ = _offline_inputs(batch)
    labels = _fixed_relations(batch, alive, groups, query_spec, seed)
    with torch.no_grad():
        context = teacher(obs, padding, alive, groups)["relation_context"]
    queries = labels["relation_queries"]
    context = context.unsqueeze(-2).expand(*queries.shape[:-1], -1)
    logits = teacher.relation_head(torch.cat((context, queries), dim=-1)).squeeze(-1)
    mask = labels["relation_mask"]
    return logits[mask], labels["relation_labels"][mask], queries[..., :3][mask].argmax(-1)


def fit_relations(bundle, data, output, steps=200, batch_size=16):
    """Fit the shared experimental relation probe on raw contexts only."""
    if min(steps, batch_size) < 1:
        raise ValueError("Positive relation-probe step and batch counts are required")
    bundle, data, output = Path(bundle), Path(data), Path(output)
    manifest = load_bundle(bundle)
    teacher_hash = manifest["artifacts"]["teacher.pth"]
    train = load_offline_dataset(data, "offline_train", teacher_hash)
    valid = load_offline_dataset(data, "offline_validation", teacher_hash)
    teacher = AgentGroupConfig(**manifest["teacher"]).get_agent_group().eval()
    teacher.load_state_dict(_load_weights(bundle / "teacher.pth"), strict=True)
    teacher.requires_grad_(False)
    teacher.relation_head.requires_grad_(True)
    query_spec = _query_spec(manifest["student_config"])
    seed = manifest["seed"]
    optimizer = torch.optim.Adam(teacher.relation_head.parameters(), lr=1e-3)
    started = perf_counter()
    train_count, train_pos = torch.zeros(3), torch.zeros(3)
    for step in range(steps):
        batch = _offline_batch(train, batch_size, seed + step)
        logits, labels, kind = _probe_batch(teacher, batch, query_spec, seed + step)
        if not labels.numel():
            raise ValueError("Relation probe received no valid training queries")
        train_count += kind.bincount(minlength=3)
        train_pos.scatter_add_(0, kind, labels)
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.binary_cross_entropy_with_logits(logits, labels).backward()
        optimizer.step()

    prior = train_pos / train_count.clamp_min(1)
    count, positives = torch.zeros(3), torch.zeros(3)
    error, prior_error = torch.zeros(3), torch.zeros(3)
    with torch.no_grad():
        for step, batch in enumerate(TrajectoryDataLoader(valid, batch_size, shuffle=False)):
            logits, labels, kind = _probe_batch(teacher, batch, query_spec, seed + 100000 + step)
            count += kind.bincount(minlength=3)
            positives.scatter_add_(0, kind, labels)
            error.scatter_add_(0, kind, (logits.sigmoid() - labels).square())
            prior_error.scatter_add_(0, kind, (prior[kind] - labels).square())

    enough_labels = (train_pos > 0) & (train_pos < train_count) & (positives > 0) & (positives < count)
    report = {
        "experimental": True,
        "teacher_sha256": teacher_hash,
        "data_manifest_sha256": _digest(data / "manifest.yaml"),
        "query_spec": query_spec,
        "steps": steps,
        "fit_and_validation_seconds": perf_counter() - started,
        "batch_size": batch_size,
        "train_positive": train_pos.tolist(),
        "train_count": train_count.tolist(),
        "validation_positive": positives.tolist(),
        "validation_count": count.tolist(),
        "validation_brier": (error / count.clamp_min(1)).tolist(),
        "validation_prior_brier": (prior_error / count.clamp_min(1)).tolist(),
        "eligible": bool(enough_labels.all() and error.sum() < prior_error.sum()),
    }
    output.mkdir(parents=True, exist_ok=False)
    torch.save(teacher.relation_head.state_dict(), output / "relation_head.pth")
    report["probe_sha256"] = _digest(output / "relation_head.pth")
    _write_config(output / "manifest.yaml", report)
    return report


def _write_config(path, content):
    with path.open("w") as stream:
        yaml.safe_dump(content, stream, sort_keys=False)


@torch.no_grad()
def _offline_validation(curriculum, batch, query_spec, seed):
    obs, padding, alive, groups, available = _offline_inputs(batch)
    teacher = curriculum.teacher
    raw = teacher(obs, padding, alive, groups)
    student = curriculum.student(obs, padding, alive, groups)
    mixed = teacher(
        obs, padding, alive, groups, message_override=student["decoded_tokens"],
        replace_mask=alive,
    )
    key = curriculum.student.output_key
    metrics = {"windows": len(obs), "live_agents": int(alive.sum())}
    labels = None if query_spec is None else _fixed_relations(batch, alive, groups, query_spec, seed)
    for name, model, output in (("codec", teacher, mixed), ("student", curriculum.student, student)):
        metrics[name] = {
            "action_kl" if key == "action_logits" else "action_q_mse": float(
                action_distillation(output[key], raw[key], alive, key, available)
            )
        }
        if labels is not None:
            metrics[name]["relation_loss"] = float(model.relation_loss(output, labels))
    return metrics


def warmup(bundle, data, output, replacement, stage_steps, readout_steps, batch_size,
           probe=None, kl_only=False, resume=False):
    """Run paired hard-message distillation followed by a frozen-codec handoff."""
    if min(readout_steps, batch_size) < 1:
        raise ValueError("Positive readout step and batch counts are required")
    if (probe is not None) == kl_only:
        raise ValueError("Supply an eligible relation probe or explicitly select KL-only smoke")
    bundle, data, output = Path(bundle), Path(data), Path(output)
    manifest = load_bundle(bundle)
    teacher_hash = manifest["artifacts"]["teacher.pth"]
    train = load_offline_dataset(data, "offline_train", teacher_hash)
    validation = load_offline_dataset(data, "offline_validation", teacher_hash)
    validation_batch = _offline_batch(validation, 128, manifest["seed"] + 300000)
    data_hash = _digest(data / "manifest.yaml")
    student = AgentGroupConfig(**_agent_spec(manifest["student_config"])).get_agent_group()
    student.load_state_dict(_load_weights(bundle / "actor_init.pth"), strict=True)
    teacher = AgentGroupConfig(**manifest["teacher"]).get_agent_group()
    teacher.load_state_dict(_load_weights(bundle / "teacher.pth"), strict=True)
    probe_hash = None
    if probe is not None:
        probe = Path(probe)
        probe_report = _read_config(probe / "manifest.yaml")
        probe_hash = _digest(probe / "relation_head.pth")
        if (
            not probe_report["eligible"]
            or probe_report["teacher_sha256"] != teacher_hash
            or probe_report["data_manifest_sha256"] != data_hash
            or probe_report["probe_sha256"] != probe_hash
        ):
            raise ValueError("Relation probe is ineligible or belongs to another teacher/data set")
        teacher.relation_head.load_state_dict(_load_weights(probe / "relation_head.pth"))
    query_spec = None if kl_only else probe_report["query_spec"]
    curriculum = CommunicationCurriculum(student, teacher, tuple(stage_steps), replacement)
    codec_parameters = list(student.codec.parameters())
    front_parameters = list(student.features.parameters()) + list(student.history.parameters())
    frozen_ids = {id(p) for p in codec_parameters + front_parameters}
    codec_optimizer = torch.optim.Adam(codec_parameters, lr=1e-3)
    readout_optimizer = torch.optim.Adam(
        [p for p in student.parameters() if id(p) not in frozen_ids], lr=1e-3
    )
    settings = {
        "experimental": True,
        "seed": manifest["seed"],
        "bundle_manifest_sha256": _digest(bundle / "manifest.yaml"),
        "data_manifest_sha256": data_hash,
        "probe_sha256": probe_hash,
        "kl_only": kl_only,
        "replacement": replacement,
        "stage_steps": list(stage_steps),
        "readout_steps": readout_steps,
        "batch_size": batch_size,
    }
    state_path = output / "offline_state.pth"
    losses = []
    diagnostics = {}
    prior_seconds = 0.0
    if resume:
        state = _load_weights(state_path)
        if state["settings"] != settings:
            raise ValueError("Offline resume settings or input hashes differ")
        curriculum.load_state_dict(state["curriculum"])
        codec_optimizer.load_state_dict(state["codec_optimizer"])
        readout_optimizer.load_state_dict(state["readout_optimizer"])
        losses = state["losses"]
        diagnostics = state.get("validation", {})
        prior_seconds = state.get("offline_seconds", 0.0)
    else:
        output.mkdir(parents=True, exist_ok=False)

    seed = manifest["seed"]
    codec_steps = sum(stage_steps)
    total = codec_steps + readout_steps
    student.requires_grad_(False)
    student.codec.requires_grad_(True)
    started = perf_counter()
    start = len(losses)
    if start == 0:
        diagnostics["initial"] = _offline_validation(curriculum, validation_batch, query_spec, seed)
    boundaries = (stage_steps[0], sum(stage_steps[:2]), codec_steps, total)
    for step in range(start, total):
        batch = _offline_batch(train, batch_size, seed + step)
        args = _offline_inputs(batch)
        labels = (
            None if kl_only else _fixed_relations(batch, args[2], args[3], query_spec, seed + step)
        )
        if step < codec_steps:
            # Batch order and labels never consume the replacement-mask RNG stream.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed + 200000 + step)
                loss = curriculum.step(codec_optimizer, *args, batch=labels)
        else:
            if step in (codec_steps, start):
                curriculum.zero_grad(set_to_none=True)
                for parameter in student.parameters():
                    parameter.requires_grad_(id(parameter) not in frozen_ids)
            loss = curriculum.readout_step(readout_optimizer, *args, batch=labels)
        if not math.isfinite(loss):
            raise RuntimeError("Non-finite CP04 offline loss")
        losses.append(loss)
        if step + 1 in (codec_steps, total):
            phase = "after_codec" if step + 1 == codec_steps else "after_readout"
            diagnostics[phase] = _offline_validation(curriculum, validation_batch, query_spec, seed)
        if step + 1 in boundaries or (step + 1) % 25 == 0:
            temporary = state_path.with_suffix(".tmp")
            torch.save({
                "settings": settings,
                "curriculum": curriculum.state_dict(),
                "codec_optimizer": codec_optimizer.state_dict(),
                "readout_optimizer": readout_optimizer.state_dict(),
                "losses": losses,
                "validation": diagnostics,
                "offline_seconds": prior_seconds + perf_counter() - started,
            }, temporary)
            temporary.replace(state_path)

    actor_path = output / "actor.pth"
    torch.save(student.state_dict(), actor_path)
    _write_config(output / "manifest.yaml", {
        **settings,
        "actor_sha256": _digest(actor_path),
        "codec_updates": int(curriculum.updates),
        "readout_updates": int(curriculum.readout_updates),
        "query_spec": query_spec,
        "validation": diagnostics,
        "offline_seconds": prior_seconds + perf_counter() - started,
    })


def _online_seed_schedule(config, rounds, splits):
    """Mirror MARLite's seeded manager streams and reserve all held-out maps."""
    rollout = config["rollout"]
    seed = config["trainer"]["seed"]
    mappo = config["trainer"]["type"] == "MAPPO"

    def schedule(stream, count, n_episodes):
        return [
            [derive_seed(derive_seed(seed, stream, index), 0, j) for j in range(n_episodes)]
            for index in range(count)
        ]

    train = schedule(ROLLOUT_STREAM, rounds + int(mappo), rollout["n_episodes"])
    diagnostics = (
        [] if mappo else schedule(EVALUATION_STREAM, rounds, rollout.get("n_eval_episodes", 10))
    )
    maps = [value for batch in train + diagnostics for value in batch]
    reserved = _DIAGNOSTIC_SEEDS.union(*(set(values) for values in splits.values()))
    if len(maps) != len(set(maps)) or reserved.intersection(maps):
        raise ValueError("Online rollout streams collide with each other or reserved maps")
    return {"collection": train, "trainer_diagnostics": diagnostics}


def run_online(
    bundle, data, output, rounds, device, distillation_steps=0,
    offline=None, probe=None, kl_only_smoke=False, structured=False, paper_baseline=None,
    environment_steps=None,
):
    """Start fresh quantized training from the common or warmed student weights."""
    if type(rounds) is not int or rounds < 1 or distillation_steps < 0:
        raise ValueError("Invalid online round or distillation budget")
    if (structured or paper_baseline is not None) and (offline is not None or distillation_steps):
        raise ValueError("The fixed or paper controls do not use CP04 warm-up or teacher distillation")
    if paper_baseline is not None and (
        paper_baseline not in CP04_BASELINES or structured or probe is not None or kl_only_smoke
    ):
        raise ValueError("Select one paper control with its own auxiliary objective")
    if kl_only_smoke and rounds != 1:
        raise ValueError("KL-only wiring smoke is limited to one online round")
    if environment_steps is not None and (
        type(environment_steps) is not int or environment_steps < 1
    ):
        raise ValueError("The training-step budget must be a positive integer")
    bundle, data, output = (Path(path).resolve() for path in (bundle, data, output))
    manifest = load_bundle(bundle)
    data_manifest = _read_config(data / "manifest.yaml")
    data_hash = _digest(data / "manifest.yaml")
    teacher_hash = manifest["artifacts"]["teacher.pth"]
    if data_manifest["teacher_sha256"] != teacher_hash:
        raise ValueError("Online run data do not belong to this teacher")
    config = deepcopy(manifest["student_config"])
    agent = config["agent_group"]
    trainer_spec = config["trainer"]
    variant = "structured_pursuit" if structured else "quant_ppo"
    if structured:
        _query_spec(config)
        agent["experimental_message_codec"] = "structured_pursuit"
    if paper_baseline is not None:
        _query_spec(config)
        variant = f"adapted_{paper_baseline}"
        agent["experimental_cp04_baseline"] = paper_baseline
        agent["auxiliary_weight"] = 0.1
        if paper_baseline == "masia" and trainer_spec.get("n_steps", 1) != 1:
            raise ValueError("The MASIA adaptation uses a one-step forecast target")
    actor_path = bundle / "actor_init.pth"
    warm = None
    if offline is not None:
        offline = Path(offline)
        warm = _read_config(offline / "manifest.yaml")
        actor_path = offline / "actor.pth"
        if (
            warm["bundle_manifest_sha256"] != _digest(bundle / "manifest.yaml")
            or warm["data_manifest_sha256"] != data_hash
            or warm["actor_sha256"] != _digest(actor_path)
            or warm["kl_only"] != kl_only_smoke
        ):
            raise ValueError("Incompatible CP04 offline handoff")
        variant = "distill_all" if warm["replacement"] == "all" else "distill_curriculum"
        agent.update(
            teacher_checkpoint=str(bundle / "teacher.pth"),
            teacher_config=manifest["teacher"],
            distillation_steps=distillation_steps,
            distillation_weight=0.1,
        )
    elif distillation_steps:
        raise ValueError("The ordinary quantization control must not distill online")

    if kl_only_smoke or paper_baseline is not None:
        agent["relation_targets"] = None
    else:
        if probe is None:
            raise ValueError("Joint online training requires the shared eligible relation probe")
        probe = Path(probe)
        relation = _read_config(probe / "manifest.yaml")
        if (
            not relation["eligible"]
            or relation["teacher_sha256"] != teacher_hash
            or relation["data_manifest_sha256"] != data_hash
            or relation["probe_sha256"] != _digest(probe / "relation_head.pth")
            or (warm is not None and warm["probe_sha256"] != relation["probe_sha256"])
        ):
            raise ValueError("Ineligible or incompatible relation probe for online training")
        agent["relation_targets"] = {"type": "ComplementaryAttackQueries", **relation["query_spec"]}
        agent["auxiliary_weight"] = 0.1

    trainer_spec.update(workdir=str(output), train_device=device, n_workers=0)
    rollout = config["rollout"]
    rollout["n_workers"] = 1
    if min(rollout["n_episodes"], rollout["episode_limit"]) < 1:
        raise ValueError("CP04 requires positive rollout episode counts and limits")
    if environment_steps is not None:
        if (rollout["manager_type"], rollout["worker_type"]) != ("persistent-env", "persistent-env"):
            raise ValueError("Exact CP04 step budgets require the native persistent rollout")
        if rounds * rollout["n_episodes"] * rollout["episode_limit"] < environment_steps:
            raise ValueError("Round limit is too small for the requested training-step budget")
        # Interrupted episodes must not hide interactions spent on a retry.
        rollout["env_retry"] = {**(rollout.get("env_retry") or {}), "max_retries": 0}
    train_args = trainer_spec["train_args"]
    train_args["target_first_metric"] = 1.1
    if manifest["algorithm"] == "MAPPO":
        train_args["iterations"] = rounds
        trainer_spec["experimental_message_target_kl"] = 0.02
    else:
        train_args["epochs"] = rounds
        train_args["rollback_interval"] = rounds + 1
        trainer_spec["eval_episodes_to_replay_ratio"] = 0.0
    schedule = _online_seed_schedule(config, rounds, data_manifest["splits"])

    output.mkdir(parents=True, exist_ok=False)
    _write_config(output / "config.yaml", config)
    run = {
        "experimental": True,
        "variant": variant,
        "kl_only_smoke": kl_only_smoke,
        "bundle_manifest_sha256": _digest(bundle / "manifest.yaml"),
        "data_manifest_sha256": data_hash,
        "initial_actor_sha256": _digest(actor_path),
        "planned_online_maps": schedule,
        "environment_step_budget": environment_steps,
        "complete": False,
    }
    _write_config(output / "manifest.yaml", run)
    trainer_config = TrainerConfig(config)
    trainer = trainer_config.create_trainer()
    initial = _load_weights(actor_path)
    if structured:
        # Only the fixed record encoder replaces learned weights; reuse the decoder.
        for name in ("codec.encoder.weight", "codec.encoder.bias"):
            del initial[name]
    if paper_baseline is not None:
        initial.update(
            (key, value) for key, value in get_state_dict(trainer.eval_agent_group).items()
            if key.startswith("cp04_baseline.")
        )
    initialize_online(trainer, initial, _load_weights(bundle / "critic_init.pth"))
    run["actor_parameters"] = sum(p.numel() for p in trainer.eval_agent_group.parameters())
    run["trainable_actor_parameters"] = sum(
        p.numel() for p in trainer.eval_agent_group.parameters() if p.requires_grad
    )
    audit = AuditedRollouts(trainer.rolloutmanager_config, environment_steps)
    trainer.rolloutmanager_config = audit
    updates = dict(actor=0, critic=0)

    def count_updates(name):
        def increment(*_):
            updates[name] += 1
        return increment

    hooks = [
        trainer.agent_optimizer.register_step_post_hook(count_updates("actor")),
        trainer.critic_optimizer.register_step_post_hook(count_updates("critic")),
    ]
    start = perf_counter()
    try:
        trainer.train(**trainer_config.train_args)
        if environment_steps is not None and not audit.collection_exhausted:
            raise RuntimeError("Round limit ended before the exact CP04 step budget was reached")
        if not updates["actor"]:
            raise RuntimeError("CP04 online training made no actor updates")
        trainer.save_current_model("final")
        run["final_actor_sha256"] = _digest(output / "checkpoints/final/agent/agent.pth")
        run["complete"] = True
    finally:
        for hook in hooks:
            hook.remove()
        run.update(
            collection_environment_steps=audit.collection_steps,
            trainer_diagnostic_environment_steps=audit.diagnostic_steps,
            actual_rollouts=audit.batches,
            training_diagnostics=audit.training_diagnostics,
            optimizer_steps=updates,
            online_seconds=perf_counter() - start,
        )
        _write_config(output / "manifest.yaml", run)
    return run


def evaluate_checkpoint(
    run_directory, data, checkpoint, split, output, locked_sha256=None, *, greedy=False,
):
    """Evaluate frozen weights on an explicit held-out split without replay."""
    if split not in ("online_validation", "test"):
        raise ValueError("Evaluation must select online_validation or test")
    run_directory, data, checkpoint, output = map(Path, (run_directory, data, checkpoint, output))
    run = _read_config(run_directory / "manifest.yaml")
    if not run["complete"] or _digest(data / "manifest.yaml") != run["data_manifest_sha256"]:
        raise ValueError("Evaluation requires a completed run and its original split manifest")
    checkpoints = run_directory.resolve() / "checkpoints"
    if not checkpoint.resolve().is_relative_to(checkpoints):
        raise ValueError("Checkpoint is outside the requested CP04 run")
    digest = _digest(checkpoint)
    final_actor = checkpoints / "final/agent/agent.pth"
    if checkpoint.resolve() == final_actor and digest != run["final_actor_sha256"]:
        raise ValueError("Final CP04 checkpoint hash changed")
    if split == "test" and (run["kl_only_smoke"] or locked_sha256 != digest):
        raise ValueError("Final test requires a locked non-smoke checkpoint SHA-256")
    config = _read_config(run_directory / "config.yaml")
    spec = _agent_spec(config)
    for key in ("teacher_checkpoint", "teacher_config", "distillation_steps", "distillation_weight"):
        spec.pop(key, None)
    state = {key: value for key, value in _load_weights(checkpoint).items()
             if not key.startswith("teacher.")}
    seeds = _read_config(data / "manifest.yaml")["splits"][split]
    return _evaluate_actor(config, spec, state, seeds, split, digest, output, greedy=greedy)


def evaluate_reference(
    bundle, data, split, output, locked_sha256=None,
    local_config=None, local_actor=None, *, greedy=False,
):
    """Evaluate the archived raw teacher or a matching local-only CP03 reference."""
    if split not in ("online_validation", "test"):
        raise ValueError("Evaluation must select online_validation or test")
    bundle, data, output = map(Path, (bundle, data, output))
    manifest = load_bundle(bundle)
    splits = _read_config(data / "manifest.yaml")
    if splits["teacher_sha256"] != manifest["artifacts"]["teacher.pth"]:
        raise ValueError("Reference does not match the split's raw teacher")
    config = manifest["student_config"]
    spec = deepcopy(manifest["teacher"])
    checkpoint = bundle / "teacher.pth"
    if local_config is not None or local_actor is not None:
        if local_config is None or local_actor is None:
            raise ValueError("Supply both local config and local actor checkpoint")
        local = _read_config(local_config)
        local_spec = _agent_spec(local)
        legacy, stock = _ALGORITHMS[manifest["algorithm"]]
        if local_spec["type"] not in (legacy, stock) or local_spec.get("message_mode") != "local":
            raise ValueError("Expected a matching CP03 local-only reference")
        local_spec["type"] = stock
        for key in _GUIDANCE_FIELDS:
            local_spec.pop(key, None)
        spec["message_mode"] = "local"
        if (
            local_spec != spec
            or local["trainer"]["type"] != manifest["algorithm"]
            or local["environment"] != config["environment"]
            or local["rollout"]["traj_len"] != config["rollout"]["traj_len"]
        ):
            raise ValueError("Local reference does not match the paired raw architecture/environment")
        checkpoint = Path(local_actor)
    digest = _digest(checkpoint)
    if split == "test" and locked_sha256 != digest:
        raise ValueError("Final test requires a locked reference SHA-256")
    return _evaluate_actor(
        config, spec, _load_weights(checkpoint), splits["splits"][split], split, digest,
        output, greedy=greedy,
    )


def _evaluate_actor(config, spec, state, seeds, split, digest, output, *, greedy=False):
    if output.exists():
        raise FileExistsError(output)
    episodes = _rollout_actor(config, spec, state, seeds, greedy=greedy)
    if len(episodes) != len(seeds):
        raise RuntimeError("Incomplete CP04 evaluation")
    rows = [{
        "seed": seed,
        "capture_fraction": float(episode["capture_fraction"]),
        "effective_hit_rate": float(episode["effective_hit_rate"]),
        "missed_attack_rate": float(episode["missed_attack_rate"]),
        "steps": int(episode["episode_length"]),
        "communication_bytes": sum(int(value) for step in episode["communication_bytes"]
                                   for value in step.values()),
    } for seed, episode in zip(seeds, episodes, strict=True)]
    result = {
        "experimental": True,
        "checkpoint_sha256": digest,
        "split": split,
        "execution": "sample" if config["trainer"]["type"] == "MAPPO" and not greedy else "greedy",
        "episodes": rows,
        "mean_capture_fraction": sum(row["capture_fraction"] for row in rows) / len(rows),
    }
    with output.open("x") as stream:
        yaml.safe_dump(result, stream, sort_keys=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preparation = subparsers.add_parser("prepare", help="Create a paired initialization bundle")
    preparation.add_argument("--source-config", required=True, type=Path)
    preparation.add_argument("--source-actor", required=True, type=Path)
    preparation.add_argument("--output", required=True, type=Path)
    preparation.add_argument("--seed", required=True, type=int)
    collection = subparsers.add_parser("collect", help="Collect disjoint offline teacher data")
    collection.add_argument("--bundle", required=True, type=Path)
    collection.add_argument("--split-file", required=True, type=Path)
    collection.add_argument("--output", required=True, type=Path)
    probe = subparsers.add_parser("fit-relations", help="Fit one shared raw relation probe")
    offline = subparsers.add_parser("warmup", help="Train codec and hand off the student readout")
    for stage in (probe, offline):
        stage.add_argument("--bundle", required=True, type=Path)
        stage.add_argument("--data", required=True, type=Path)
        stage.add_argument("--output", required=True, type=Path)
        stage.add_argument("--batch-size", type=int, default=16)
    probe.add_argument("--steps", type=int, default=200)
    offline.add_argument("--replacement", choices=("all", "progressive"), required=True)
    offline.add_argument("--stage-steps", type=int, nargs=3, default=(200, 200, 200))
    offline.add_argument("--readout-steps", type=int, default=200)
    supervision = offline.add_mutually_exclusive_group(required=True)
    supervision.add_argument("--probe", type=Path)
    supervision.add_argument("--kl-only", action="store_true")
    offline.add_argument("--resume", action="store_true")
    online = subparsers.add_parser("online", help="Start fresh fully quantized training")
    for key in ("bundle", "data", "output"):
        online.add_argument(f"--{key}", required=True, type=Path)
    online.add_argument("--rounds", required=True, type=int)
    online.add_argument("--environment-steps", type=int)
    online.add_argument("--device", default="cpu")
    online.add_argument("--distillation-steps", type=int, default=0)
    online.add_argument("--offline", type=Path)
    online.add_argument("--probe", type=Path)
    online.add_argument("--kl-only-smoke", action="store_true")
    online.add_argument("--structured", action="store_true")
    online.add_argument("--paper-baseline", choices=CP04_BASELINES)
    evaluation = subparsers.add_parser("evaluate", help="Evaluate on a fixed held-out split")
    for key in ("run-directory", "data", "checkpoint", "output"):
        evaluation.add_argument(f"--{key}", required=True, type=Path)
    evaluation.add_argument("--split", choices=("online_validation", "test"), required=True)
    evaluation.add_argument("--locked-sha256")
    evaluation.add_argument("--greedy", action="store_true")
    reference = subparsers.add_parser("reference", help="Evaluate a frozen raw/local reference")
    for key in ("bundle", "data", "output"):
        reference.add_argument(f"--{key}", required=True, type=Path)
    reference.add_argument("--split", choices=("online_validation", "test"), required=True)
    reference.add_argument("--locked-sha256")
    reference.add_argument("--greedy", action="store_true")
    reference.add_argument("--local-config", type=Path)
    reference.add_argument("--local-actor", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source_config, args.source_actor, args.output, args.seed)
    elif args.command == "collect":
        collect(args.bundle, args.split_file, args.output)
    elif args.command == "fit-relations":
        report = fit_relations(args.bundle, args.data, args.output, args.steps, args.batch_size)
        print(f"Relation probe eligible: {report['eligible']}")
    elif args.command == "warmup":
        warmup(
            args.bundle, args.data, args.output, args.replacement,
            args.stage_steps, args.readout_steps, args.batch_size,
            args.probe, args.kl_only, args.resume,
        )
    elif args.command == "online":
        run_online(
            args.bundle, args.data, args.output, args.rounds, args.device,
            args.distillation_steps, args.offline, args.probe, args.kl_only_smoke,
            args.structured, args.paper_baseline, args.environment_steps,
        )
    elif args.command == "evaluate":
        evaluate_checkpoint(
            args.run_directory, args.data, args.checkpoint, args.split,
            args.output, args.locked_sha256, greedy=args.greedy,
        )
    else:
        evaluate_reference(
            args.bundle, args.data, args.split, args.output, args.locked_sha256,
            args.local_config, args.local_actor, greedy=args.greedy,
        )


if __name__ == "__main__":
    main()
