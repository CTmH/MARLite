"""Python-owned recipes, independent of the component registry's constructors."""

from collections.abc import Mapping
from copy import deepcopy
import inspect
import math
from pathlib import Path

from marlite.collection.parallel import ParallelCollector
from marlite.communication.context import AllTogether
from marlite.communication.group_set import QuantizedCommunication
from marlite.data.replay import EpisodeReplay
from marlite.learning.auxiliary import AuxiliaryLearning
from marlite.learning.distillation import (
    FrozenPolicyReference,
    CompositeObjective,
    ActionDistillationObjective,
)
from marlite.learning.mappo import MAPPOLearner
from marlite.learning.schedules import OptimizerSchedules, ScheduleBinding
from marlite.learning.value import ValueLearner
from marlite.training.mappo import SequenceMAPPOTrainer
from marlite.training.value import SequenceValueTrainer
from marlite.training.distillation import SequenceDistillationTrainer
from marlite.training.distributed import initialize_backend
from marlite.util.randomness import configure_randomness, preserve_rng_state
from .catalog import default_registry
from .environment import EnvironmentFactory, EnvironmentSpec
from .registry import ConfigurationError
from . import defaults


RECIPES = ("mappo", "qmix", "qplex", "qtran", "qmpd")


def _mapping(value, path, allowed=None):
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{path}: expected a mapping")
    if allowed is not None and value.keys() - allowed:
        raise ConfigurationError(f"{path}: unknown fields {sorted(value.keys() - allowed)}")
    return deepcopy(dict(value))


def _construct(factory, parameters, path):
    try:
        inspect.signature(factory).bind(**parameters)
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"{path}: {error}") from error
    try:
        return factory(**parameters)
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"{path}: {error}") from error


def optimizer(registry, spec, module, dimensions, *, lr=0.0005, path="optimizer"):
    spec = deepcopy(spec or {"type": "Adam", "params": {"lr": lr}})
    groups = spec.pop("groups", [])
    if not isinstance(groups, list):
        raise ConfigurationError(f"{path}.groups: expected a list")
    named = {
        name: parameter for name, parameter in module.named_parameters() if parameter.requires_grad
    }
    if not named:
        if groups:
            raise ConfigurationError(f"{path}: no parameters for optimizer groups")
        return None
    remaining, parameters = dict(named), []
    for index, group in enumerate(groups):
        group = _mapping(group, f"{path}.groups[{index}]", {"prefixes", "params"})
        prefixes = group.get("prefixes", [])
        if (
            not isinstance(prefixes, list)
            or not prefixes
            or any(not isinstance(prefix, str) for prefix in prefixes)
        ):
            raise ConfigurationError(f"{path}: groups require nonempty prefix lists")
        selected = {}
        for prefix in prefixes:
            matches = {
                name: parameter
                for name, parameter in named.items()
                if name == prefix or name.startswith(prefix + ".")
            }
            if not matches or not matches.keys() <= remaining.keys():
                raise ConfigurationError(
                    f"{path}: unmatched or overlapping parameter prefix {prefix!r}"
                )
            selected.update(matches)
        options = _mapping(group.get("params", {}), path + ".group.params")
        if "params" in options:
            raise ConfigurationError(f"{path}: parameter tensors cannot be specified in YAML")
        parameters.append({"params": list(selected.values()), **options})
        for name in selected:
            remaining.pop(name)
    if remaining:
        parameters.append({"params": list(remaining.values())})
    return registry.build(spec, dimensions, path=path, injected={"params": parameters})


def schedules(registry, specifications, optimizers, dimensions, *, path="schedulers"):
    """Attach ordinary Torch schedulers; Python owns optimizer/event wiring."""
    specifications = _mapping(specifications, path, set(optimizers))
    bindings = {}
    for role, specification in specifications.items():
        entry = _mapping(specification, path + "." + role, {"scheduler", "event", "metric"})
        if optimizers[role] is None:
            raise ConfigurationError(f"{path}.{role}: optimizer is absent")
        if "scheduler" not in entry:
            raise ConfigurationError(f"{path}.{role}: scheduler specification is required")
        scheduler = registry.build(
            entry.pop("scheduler"),
            dimensions,
            path=path + "." + role + ".scheduler",
            injected={"optimizer": optimizers[role]},
        )
        bindings[role] = _construct(
            ScheduleBinding, {"scheduler": scheduler, **entry}, path + "." + role
        )
    return OptimizerSchedules(bindings)


class Experiment:
    """A built experiment; teacher preparation is explicit and runs only on run()."""

    def __init__(
        self, config, trainer, environment_spec, registry, teacher_trainer=None, teacher_run=None
    ):
        self.config, self.trainer, self.environment_spec = config, trainer, environment_spec
        self.registry, self.teacher_trainer, self.teacher_run = (
            registry,
            teacher_trainer,
            teacher_run,
        )
        self.teacher_prepared = teacher_trainer is None

    def run(self, *, resume=None, run_options=None):
        from .metadata import save_experiment_metadata

        trainer = self.trainer
        options = self.config.get("run", {}) if run_options is None else dict(run_options)
        try:
            inspect.signature(trainer.run).bind(**options)
        except TypeError as error:
            raise ConfigurationError(f"run: {error}") from error
        for name in ("iterations", "train_episodes", "eval_episodes"):
            if name in options:
                minimum = 0 if name == "iterations" else 1
                if type(options[name]) is not int or options[name] < minimum:
                    raise ConfigurationError(f"run.{name}: expected an integer >= {minimum}")
        if resume is not None:
            trainer.load_checkpoint(Path(resume).expanduser())
            self.teacher_prepared = True
        elif (
            trainer.workdir is not None
            and not trainer.iterations
            and not any(getattr(trainer, "offline_counters", {}).values())
        ):
            if any(
                (trainer.workdir / name).exists()
                for name in ("events.jsonl", "offline_events.jsonl")
            ):
                raise FileExistsError(
                    "This workdir already contains a run; resume a checkpoint or choose another directory"
                )
        self.config["run"] = options
        save_experiment_metadata(self)
        if not self.teacher_prepared:
            self.teacher_trainer.run(**self.teacher_run)
            trainer.adopt_teacher(
                self.teacher_trainer.learner.actor.state_dict(),
                self.teacher_trainer.learner.critic.state_dict(),
            )
            self.teacher_prepared = True
        trainer.run(**options)
        if trainer.workdir is not None:
            trainer.save_checkpoint(trainer.workdir / "checkpoints" / "final.pt")
        return trainer


def _probe_environment(factory, backend, seed):
    # Only rank 0 starts simulators; other learners need just the small contract.
    # This avoids launching one StarCraft/SUMO instance per GPU at startup.
    result = None
    if backend.rank == 0:
        try:
            devices = [backend.device.index] if backend.device.type == "cuda" else []
            with preserve_rng_state(cuda_devices=devices):
                result = {"spec": factory.probe(seed).state_dict()}
        except Exception as error:
            result = {"error": repr(error)}
    result = backend.broadcast_object(result)
    if "error" in result:
        raise ConfigurationError("environment probe failed: " + result["error"])
    spec = result["spec"]
    return EnvironmentSpec(
        tuple(spec["agent_ids"]),
        tuple(spec["observation_shape"]),
        tuple(spec["state_shape"]),
        spec["action_dim"],
    )


def build_experiment(
    config, *, workdir=None, device="cpu", registry=None, backend=None, policy_only=False
):
    """Build an experiment from a version-3 configuration mapping.

    Pass an extended registry to make custom components available. The device
    selects the learner device and workdir selects the run output directory.
    Environment probing resolves model dimensions during construction. Call
    Experiment.run to start training. With policy_only enabled, the function
    returns a PolicyEvaluator for loading and evaluating actor weights.
    """
    allowed = {
        "version",
        "recipe",
        "seed",
        "deterministic",
        "environment",
        "evaluation_environment",
        "dimensions",
        "defaults",
        "actor",
        "critic",
        "objective",
        "auxiliary",
        "optimizers",
        "schedulers",
        "communication_context",
        "collector",
        "learner",
        "trainer",
        "replay",
        "run",
        "teacher",
        "offline",
    }
    config = _mapping(config, "config", allowed)
    if policy_only:

        def without_checkpoints(value):
            if isinstance(value, dict):
                return {
                    key: without_checkpoints(item)
                    for key, item in value.items()
                    if not (key == "checkpoint" and "type" in value)
                }
            if isinstance(value, list):
                return [without_checkpoints(item) for item in value]
            return value

        config = without_checkpoints(config)
    if (
        type(config.get("version")) is not int
        or config.get("version") != 3
        or config.get("recipe") not in RECIPES
    ):
        raise ConfigurationError(
            "Use version: 3 and a supported recipe; old configurations are intentionally incompatible"
        )
    registry = registry or default_registry()
    recipe = config["recipe"]
    if recipe != "qmpd" and config.keys() & {"teacher", "offline"}:
        raise ConfigurationError("Teacher/offline settings belong only to the qmpd recipe")
    backend = backend or initialize_backend(device)
    configure_randomness(config.get("seed", 42), config.get("deterministic", False))
    factory = EnvironmentFactory(config.get("environment"), registry)
    environment_spec = _probe_environment(factory, backend, config.get("seed", 42))
    dimensions = environment_spec.dimensions
    extra_dimensions = _mapping(config.get("dimensions", {}), "dimensions")
    if dimensions.keys() & extra_dimensions.keys():
        raise ConfigurationError("Custom dimensions cannot override environment dimensions")
    if any(type(value) is not int or value < 1 for value in extra_dimensions.values()):
        raise ConfigurationError("Custom dimensions must be positive integers")
    dimensions.update(extra_dimensions)
    config = registry.resolve(config, dimensions, construct=False)
    options = _mapping(
        config.get("defaults", {}),
        "defaults",
        {"hidden_dim", "mixing_dim", "heads", "learning_rate"},
    )
    hidden, mixing, heads = (
        options.get("hidden_dim", 64),
        options.get("mixing_dim", 32),
        options.get("heads", 4),
    )
    if any(type(value) is not int or value < 1 for value in (hidden, mixing, heads)):
        raise ConfigurationError("Default network dimensions must be positive integers")
    learning_rate = options.get("learning_rate", 0.0005)
    if (
        not isinstance(learning_rate, (int, float))
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ConfigurationError("Default learning_rate must be finite and positive")
    workdir = Path(workdir).expanduser() if workdir is not None else None

    def component(key, fallback):
        return registry.build(config[key], dimensions, path=key) if key in config else fallback()

    actor = component(
        "actor",
        lambda: defaults.actor(
            environment_spec,
            hidden,
            output_kind="logits" if recipe in ("mappo", "qmpd") else "q_values",
            quantized=recipe == "qmpd",
            heads=heads,
        ),
    )
    context_spec = config.get("communication_context")
    context = (
        registry.build(context_spec, dimensions, path="communication_context")
        if context_spec
        else (AllTogether() if recipe == "qmpd" else None)
    )
    collector_options = _mapping(
        config.get("collector", {}),
        "collector",
        {"max_steps", "device", "state_snapshot_interval", "retry"},
    )
    collector = ParallelCollector(factory, context_provider=context, **collector_options)
    evaluation_collector = None
    if "evaluation_environment" in config:
        evaluation_factory = EnvironmentFactory(config["evaluation_environment"], registry)
        evaluation_spec = _probe_environment(evaluation_factory, backend, config.get("seed", 42))
        if evaluation_spec != environment_spec:
            raise ConfigurationError(
                "Evaluation environment must have the same agent/model specification"
            )
        evaluation_collector = ParallelCollector(
            evaluation_factory, context_provider=context, **collector_options
        )
    if policy_only:
        from marlite.analysis import PolicyEvaluator

        actor.to(backend.device).eval()
        backend.synchronize(actor)
        return PolicyEvaluator(
            config, actor, evaluation_collector or collector, environment_spec, backend
        )
    if recipe in ("mappo", "qmpd"):
        critic = component("critic", lambda: defaults.value(environment_spec, hidden))
        objective = None
        if "objective" in config or "replay" in config:
            raise ConfigurationError("On-policy recipes do not have a value objective or replay")
    else:
        critic = component(
            "critic",
            lambda: defaults.mixer(
                recipe, environment_spec, actor, hidden=hidden, mixing=mixing, heads=heads
            ),
        )
        objective = component(
            "objective", lambda: defaults.objective(recipe, environment_spec, hidden)
        )
    opt_specs = _mapping(
        config.get("optimizers", {}), "optimizers", {"actor", "critic", "objective"}
    )
    auxiliary_config = _mapping(
        config.get("auxiliary", {}),
        "auxiliary",
        {"objective", "optimizer", "mixer", "start_update"},
    )
    auxiliary_objective = (
        registry.build(auxiliary_config["objective"], dimensions, path="auxiliary.objective")
        if "objective" in auxiliary_config
        else None
    )
    teacher_trainer = reference = teacher_critic = student_relation = None
    teacher_run = None
    if recipe == "qmpd":
        if not isinstance(actor.communication, QuantizedCommunication):
            raise ConfigurationError("qmpd requires a quantized group communication actor")
        teacher_config = _mapping(
            config.get("teacher", {}),
            "teacher",
            {
                "source",
                "actor",
                "critic",
                "learner",
                "trainer",
                "optimizers",
                "schedulers",
                "run",
                "online_distillation",
            },
        )
        if teacher_config.get("source") not in ("train", "checkpoint"):
            raise ConfigurationError("teacher.source must be train or checkpoint")
        teacher_actor = deepcopy(actor)
        teacher_actor.communication = deepcopy(actor.communication.aggregation)
        if "actor" in teacher_config:
            teacher_actor = registry.build(
                teacher_config["actor"], dimensions, path="teacher.actor"
            )
        teacher_critic = (
            registry.build(teacher_config["critic"], dimensions, path="teacher.critic")
            if "critic" in teacher_config
            else deepcopy(critic)
        )
        if teacher_config["source"] == "checkpoint":
            if "actor" not in teacher_config or "checkpoint" not in teacher_config["actor"]:
                raise ConfigurationError("Checkpoint teachers require teacher.actor.checkpoint")
        else:
            teacher_opts = _mapping(
                teacher_config.get("optimizers", {}), "teacher.optimizers", {"actor", "critic"}
            )
            teacher_learner_options = _mapping(teacher_config.get("learner", {}), "teacher.learner")
            if teacher_learner_options.keys() & {
                "actor",
                "critic",
                "backend",
                "actor_optimizer",
                "critic_optimizer",
                "auxiliary",
            }:
                raise ConfigurationError("Teacher learner wiring belongs to the Python recipe")
            teacher_trainer_options = _mapping(teacher_config.get("trainer", {}), "teacher.trainer")
            if teacher_trainer_options.keys() & {
                "learner",
                "collector",
                "evaluation_collector",
                "seed",
                "workdir",
            }:
                raise ConfigurationError("Teacher trainer wiring belongs to the Python recipe")
            teacher_learner = _construct(
                MAPPOLearner,
                {
                    **teacher_learner_options,
                    "actor": teacher_actor,
                    "critic": teacher_critic,
                    "backend": backend,
                    "actor_optimizer": optimizer(
                        registry,
                        teacher_opts.get("actor"),
                        teacher_actor,
                        dimensions,
                        lr=learning_rate,
                    ),
                    "critic_optimizer": optimizer(
                        registry,
                        teacher_opts.get("critic"),
                        teacher_critic,
                        dimensions,
                        lr=learning_rate,
                    ),
                },
                "teacher.learner",
            )
            teacher_learner.schedules = schedules(
                registry,
                teacher_config.get("schedulers", {}),
                {
                    "actor": teacher_learner.actor_optimizer,
                    "critic": teacher_learner.critic_optimizer,
                },
                dimensions,
                path="teacher.schedulers",
            )
            teacher_trainer = _construct(
                SequenceMAPPOTrainer,
                {
                    **teacher_trainer_options,
                    "learner": teacher_learner,
                    "collector": collector,
                    "evaluation_collector": evaluation_collector,
                    "seed": config.get("seed", 42),
                    "workdir": None if workdir is None else workdir / "teacher",
                },
                "teacher.trainer",
            )
            teacher_run = _mapping(teacher_config.get("run", {}), "teacher.run")
            if type(teacher_run.get("iterations")) is not int or teacher_run["iterations"] < 1:
                raise ConfigurationError(
                    "Train-source teachers require positive teacher.run.iterations"
                )
            try:
                inspect.signature(teacher_trainer.run).bind(**teacher_run)
            except TypeError as error:
                raise ConfigurationError(f"teacher.run: {error}") from error
        reference = FrozenPolicyReference(deepcopy(teacher_actor).to(backend.device))
        teacher_critic = deepcopy(teacher_critic)
        objectives = {}
        if auxiliary_objective is not None:
            student_relation = auxiliary_objective
            objectives["relation"] = auxiliary_objective
        online = _mapping(
            teacher_config.get("online_distillation", {}),
            "teacher.online_distillation",
            {"enabled", "weight", "decay_updates"},
        )
        enabled = online.pop("enabled", False)
        if type(enabled) is not bool:
            raise ConfigurationError("Online distillation enabled must be boolean")
        if enabled:
            objectives["distillation"] = ActionDistillationObjective(reference, **online)
        auxiliary_objective = CompositeObjective(objectives) if objectives else None
    auxiliary = None
    if auxiliary_objective is not None:
        aux_optimizer = optimizer(
            registry,
            auxiliary_config.get("optimizer"),
            auxiliary_objective,
            dimensions,
            lr=learning_rate,
            path="auxiliary.optimizer",
        )
        mixer = (
            registry.build(auxiliary_config["mixer"], dimensions, path="auxiliary.mixer")
            if "mixer" in auxiliary_config
            else None
        )
        auxiliary = AuxiliaryLearning(
            auxiliary_objective,
            aux_optimizer,
            mixer=mixer,
            start_update=auxiliary_config.get("start_update", 0),
        )
    elif auxiliary_config:
        raise ConfigurationError("Auxiliary options require an auxiliary objective")
    common = {
        "actor": actor,
        "critic": critic,
        "backend": backend,
        "auxiliary": auxiliary,
        "actor_optimizer": optimizer(
            registry,
            opt_specs.get("actor"),
            actor,
            dimensions,
            lr=learning_rate,
            path="optimizers.actor",
        ),
        "critic_optimizer": optimizer(
            registry,
            opt_specs.get("critic"),
            critic,
            dimensions,
            lr=learning_rate,
            path="optimizers.critic",
        ),
    }
    learner_options = _mapping(config.get("learner", {}), "learner")
    reserved = {
        "actor",
        "critic",
        "objective",
        "backend",
        "auxiliary",
        "actor_optimizer",
        "critic_optimizer",
        "objective_optimizer",
    }
    if learner_options.keys() & reserved:
        raise ConfigurationError("Learner model/optimizer wiring belongs to the Python recipe")
    if recipe in ("mappo", "qmpd"):
        learner = _construct(MAPPOLearner, common | learner_options, "learner")
    else:
        learner = _construct(
            ValueLearner,
            common
            | learner_options
            | {
                "objective": objective,
                "objective_optimizer": optimizer(
                    registry,
                    opt_specs.get("objective"),
                    objective,
                    dimensions,
                    lr=learning_rate,
                    path="optimizers.objective",
                ),
            },
            "learner",
        )
    learner.schedules = schedules(
        registry,
        config.get("schedulers", {}),
        {
            "actor": learner.actor_optimizer,
            "critic": learner.critic_optimizer,
            "objective": getattr(learner, "objective_optimizer", None),
            "auxiliary": None if learner.auxiliary is None else learner.auxiliary.optimizer,
        },
        dimensions,
    )
    trainer_options = _mapping(config.get("trainer", {}), "trainer")
    if trainer_options.keys() & {"learner", "collector", "evaluation_collector", "workdir", "seed"}:
        raise ConfigurationError("Trainer wiring/workdir/seed belongs to the Python recipe")
    arguments = trainer_options | {
        "learner": learner,
        "collector": collector,
        "evaluation_collector": evaluation_collector,
        "workdir": workdir,
        "seed": config.get("seed", 42),
    }
    if recipe == "qmpd":
        offline = _mapping(
            config.get("offline", {}),
            "offline",
            {
                "offline_enabled",
                "train_episodes",
                "validation_episodes",
                "codec_updates",
                "policy_updates",
                "offline_batch_sequences",
                "replacement",
                "replacement_steps",
                "probe_updates",
                "codec_aux_weight",
                "policy_aux_weight",
                "initialize_from_teacher",
                "optimizers",
                "schedulers",
                "probe_objective",
            },
        )
        offline_schedules = offline.pop("schedulers", {})
        stage_optimizers = _mapping(
            offline.pop("optimizers", {}), "offline.optimizers", {"codec", "policy", "probe"}
        )
        # A separate probe architecture may be specified; otherwise use an
        # independent copy of the student's prediction task/head.
        if "probe_objective" in offline:
            offline["probe_objective"] = registry.build(
                offline["probe_objective"], dimensions, path="offline.probe_objective"
            )
        elif offline.get("codec_aux_weight", 0):
            if student_relation is None:
                raise ConfigurationError("Codec auxiliary loss requires a relation objective")
            offline["probe_objective"] = deepcopy(student_relation)
        if stage_optimizers:
            from torch import nn

            policy_modules = {
                "aggregation": actor.communication.aggregation,
                "readout": actor.readout,
                "action_head": actor.action_head,
            }
            if student_relation is not None:
                policy_modules["auxiliary"] = student_relation
            stage_modules = {
                "codec": actor.communication.codec,
                "policy": nn.ModuleDict(policy_modules),
                "probe": offline.get("probe_objective"),
            }
            for name, specification in stage_optimizers.items():
                if stage_modules[name] is None:
                    raise ConfigurationError(f"offline.optimizers.{name}: this stage has no head")
                offline[name + "_optimizer"] = optimizer(
                    registry,
                    specification,
                    stage_modules[name],
                    dimensions,
                    lr=0.001,
                    path="offline.optimizers." + name,
                )
        arguments.update(
            offline,
            reference=reference,
            teacher_critic=teacher_critic,
            student_relation=student_relation,
        )
        trainer = _construct(SequenceDistillationTrainer, arguments, "trainer")
        trainer.stage_schedules = schedules(
            registry,
            offline_schedules,
            trainer.stage_optimizers,
            dimensions,
            path="offline.schedulers",
        )
    elif recipe == "mappo":
        trainer = _construct(SequenceMAPPOTrainer, arguments, "trainer")
    else:
        replay = _construct(
            EpisodeReplay, _mapping(config.get("replay", {"capacity": 1000}), "replay"), "replay"
        )
        trainer = _construct(SequenceValueTrainer, arguments | {"replay": replay}, "trainer")
    _mapping(config.get("run", {}), "run")
    trainer.configuration = {key: value for key, value in config.items() if key != "run"}
    return Experiment(config, trainer, environment_spec, registry, teacher_trainer, teacher_run)
