"""C04 stages: frozen probe, codec distillation, policy distillation, MAPPO."""

from contextlib import contextmanager
import json
import math

import torch
from torch import nn

from marlite.communication.context import frame_context
from marlite.communication.group_set import GroupSetCommunication, QuantizedCommunication
from marlite.data import Episode, SequenceSlice, collate_sequences, partition_episode
from marlite.learning.distillation import (RelationQueryObjective, codec_distillation,
                                          action_kl_term, replacement_senders)
from marlite.util.randomness import derive_seed
from .mappo import SequenceMAPPOTrainer


def _optimizer_matches(optimizer, parameters):
    actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
    expected = {id(p) for p in parameters}
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Offline optimizer must own exactly its stage parameters")


@contextmanager
def _stage_trainability(modules, active):
    flags = {p: p.requires_grad for module in modules for p in module.parameters()}
    try:
        for parameter in flags:
            parameter.requires_grad_(False)
            parameter.grad = None
        for parameter in active:
            parameter.requires_grad_(True)
        yield
    finally:
        for parameter, value in flags.items():
            parameter.requires_grad_(value)


class SequenceDistillationTrainer(SequenceMAPPOTrainer):
    """Own stage control; the shared MAPPO learner still owns all online updates.

    Both offline stages share fixed teacher episodes; validation uses a separate
    stream and never supplies gradient samples. Local encoders stay frozen during
    offline training. Phase optimizers have independent momentum from online
    optimizers, including an optional student query head. Checkpoints embed the
    teacher, offline data, stage counters, optimizers and sampling RNG.
    """

    algorithm = "sequence_distilled_group_mappo"
    _stages = ("probe", "codec", "policy")

    def __init__(self, learner, collector, reference, teacher_critic, *,
                 offline_enabled=True, train_episodes=32, validation_episodes=16,
                 codec_updates=600, policy_updates=200, offline_batch_sequences=32,
                 replacement="all", replacement_steps=None, probe_objective=None,
                 probe_updates=0, student_relation=None, codec_aux_weight=0., policy_aux_weight=0.,
                 codec_optimizer=None, policy_optimizer=None, probe_optimizer=None,
                 initialize_from_teacher=True, **kwargs):
        super().__init__(learner, collector, **kwargs)
        if not isinstance(learner.actor.communication, QuantizedCommunication):
            raise ValueError("C04 student requires QuantizedCommunication")
        if not isinstance(reference.actor.communication, GroupSetCommunication):
            raise ValueError("C04 teacher must exchange raw group features")
        if offline_enabled and not initialize_from_teacher:
            raise ValueError("Frozen offline local features require initialization from the teacher")
        if any(type(n) is not int or n < 1 for n in (train_episodes, validation_episodes, offline_batch_sequences)):
            raise ValueError("Offline dataset and batch sizes must be positive integers")
        if any(type(n) is not int or n < 0 for n in (codec_updates, policy_updates, probe_updates)):
            raise ValueError("Offline stage lengths must be nonnegative integers")
        if any(not math.isfinite(v) or v < 0 for v in (codec_aux_weight, policy_aux_weight)):
            raise ValueError("Offline auxiliary weights must be finite and nonnegative")
        if any(objective is not None and not isinstance(objective, RelationQueryObjective)
               for objective in (probe_objective, student_relation)):
            raise ValueError("Stage query heads must use RelationQueryObjective")
        if codec_aux_weight and (probe_objective is None or not probe_updates):
            raise ValueError("Codec auxiliary guidance requires a trained frozen probe")
        if policy_aux_weight and student_relation is None:
            raise ValueError("Policy auxiliary loss requires the student's query objective")
        if (probe_objective is None) != (probe_updates == 0):
            raise ValueError("Probe objective and positive probe_updates must be configured together")
        if type(offline_enabled) is not bool or type(initialize_from_teacher) is not bool:
            raise ValueError("Stage switches must be boolean")
        owned = {id(p) for p in learner.actor.parameters()} | {id(p) for p in learner.critic.parameters()}
        if owned.intersection(id(p) for module in (reference.actor, teacher_critic) for p in module.parameters()):
            raise ValueError("Teacher and student must own independent parameters")
        if offline_enabled and (reference.actor.detach_messages or reference.actor.detach_fusion
                                or not reference.actor.rl_gate):
            raise ValueError("Codec training requires a differentiable teacher message readout")
        self.reference, self.teacher_critic = reference, teacher_critic
        backend = learner.backend
        reference.actor.to(backend.device)
        teacher_critic.requires_grad_(False).eval().to(backend.device)
        backend.synchronize(reference.actor)
        backend.synchronize(teacher_critic)
        self.reference.clear()
        if initialize_from_teacher:
            if learner.updates:
                raise ValueError("Teacher initialization requires a fresh student learner")
            self._initialize_student()
        self.offline_enabled = offline_enabled
        self.offline_train_episodes, self.offline_validation_episodes = train_episodes, validation_episodes
        self.offline_batch_sequences = offline_batch_sequences
        self.stage_steps = dict(probe=probe_updates, codec=codec_updates, policy=policy_updates)
        self.replacement = replacement
        self.replacement_boundaries = ()
        if replacement == "progressive":
            if (not isinstance(replacement_steps, (tuple, list)) or len(replacement_steps) != 3
                    or any(type(n) is not int or n < 1 for n in replacement_steps)
                    or sum(replacement_steps) != codec_updates):
                raise ValueError("Three positive replacement stages must sum to codec_updates")
            self.replacement_boundaries = (replacement_steps[0], sum(replacement_steps[:2]))
        elif replacement != "all" or replacement_steps is not None:
            raise ValueError("Use all replacements, or progressive with replacement_steps")
        self.probe_objective, self.student_relation = probe_objective, student_relation
        self.codec_aux_weight, self.policy_aux_weight = codec_aux_weight, policy_aux_weight
        if student_relation is not None:
            if learner.auxiliary is None or not any(module is student_relation for module in learner.auxiliary.objective.modules()):
                raise ValueError("Student query head must belong to the online auxiliary component")
        if probe_objective is not None:
            probe_objective.to(backend.device)
            backend.synchronize(probe_objective)
        self.stage_parameters = self._stage_parameters()
        self.stage_optimizers = {}
        for name, optimizer in (("probe", probe_optimizer), ("codec", codec_optimizer), ("policy", policy_optimizer)):
            parameters = self.stage_parameters[name]
            if self.stage_steps[name] and not parameters:
                raise ValueError(f"Offline {name} stage has no trainable parameters")
            if parameters:
                optimizer = optimizer or torch.optim.Adam(parameters, lr=.001)
                _optimizer_matches(optimizer, parameters)
            elif optimizer is not None:
                raise ValueError("Cannot configure an optimizer for an absent stage")
            self.stage_optimizers[name] = optimizer
        sampler_seed = derive_seed(self.seed, 121)
        if sampler_seed is None:
            sampler_seed = backend.broadcast_object(torch.seed() if backend.rank == 0 else None)
        self.offline_rng = torch.Generator().manual_seed(sampler_seed)
        self.phase = "probe" if offline_enabled else "online"
        self.offline_counters = dict.fromkeys(self._stages, 0)
        self.offline_train, self.offline_validation = {}, {}
        self.offline_metrics = {}

    def adopt_teacher(self, actor_state, critic_state):
        """Transfer an independently trained teacher before any student fitting."""
        if self.learner.updates or any(self.offline_counters.values()) or self.offline_train:
            raise ValueError("Teacher transfer must precede all student stages")
        self.reference.actor.load_state_dict(actor_state, strict=True)
        self.teacher_critic.load_state_dict(critic_state, strict=True)
        self.reference.clear()
        self._initialize_student()

    def _initialize_student(self):
        teacher, student = self.reference.actor, self.learner.actor
        for name in ("frame_encoder", "temporal", "message_encoder", "message_temporal",
                     "message_projection", "readout", "action_head"):
            source, target = getattr(teacher, name), getattr(student, name)
            if (source is None) != (target is None):
                raise ValueError("Teacher/student frozen local components must be compatible")
            if target is not None:
                target.load_state_dict(source.state_dict(), strict=True)
        student.communication.aggregation.load_state_dict(teacher.communication.state_dict(), strict=True)
        if (teacher.message_source != "history" or student.message_source != "history"
                or teacher.message_encoder is not None or student.message_encoder is not None
                or teacher.message_temporal is not None or student.message_temporal is not None
                or teacher.message_projection is not None or student.message_projection is not None):
            raise ValueError("C04 scalar codec consumes the frozen local history features directly")

    def _stage_parameters(self):
        actor = self.learner.actor
        policy_modules = [actor.communication.aggregation, actor.readout, actor.action_head]
        if self.student_relation is not None:
            policy_modules.append(self.student_relation)
        return {"codec": tuple(actor.communication.codec.parameters()),
                "policy": tuple(dict.fromkeys(p for module in policy_modules for p in module.parameters())),
                "probe": () if self.probe_objective is None else tuple(self.probe_objective.parameters())}

    def _collect_teacher(self, n_episodes, stream, directory):
        backend = self.learner.backend
        episodes, error = {}, None
        if backend.rank == 0:
            try:
                with self._rng_context():
                    episodes = {index: self.collector.collect_episode(
                        self.reference.actor, self.teacher_critic,
                        seed=derive_seed(self.seed, stream, index), policy_version=0,
                    ) for index in range(n_episodes)}
            except Exception as exc:
                if not backend.distributed:
                    raise
                error = repr(exc)
        error = backend.broadcast_object(error)
        if error is not None:
            raise RuntimeError("Offline teacher collection failed: " + error)
        return self._publish_episodes(episodes, directory)

    def _offline_loss(self, stage, episodes, sources, update, *, validation=False):
        backend = self.learner.backend
        empty = not sources
        if empty:
            sources = [SequenceSlice(next(iter(episodes)), 0, 1)]
        batch = collate_sequences(episodes, sources).to(backend.device)
        if empty:
            batch.loss_mask.zero_()
        length = batch.loss_mask.shape[1]
        alive = batch.frames["alive"][:, :length]
        context = frame_context(batch.frames, stop=length)
        if stage == "codec":
            level = 2 if validation else sum(update >= end for end in self.replacement_boundaries)
            selected = replacement_senders(batch, context["groups"], alive, mode=self.replacement,
                                            stage=level, update=0 if validation else update, seed=self.seed or 0)
            output, term = codec_distillation(self.reference, self.learner.actor.communication.codec,
                                               batch, episodes, selected)
            terms = {"distillation_loss": term}
            if self.codec_aux_weight:
                extra = self.probe_objective(output, batch, 0 if validation else update,
                                              weight_override=self.codec_aux_weight)
                terms.update(extra)
        else:
            cached = self.reference.batch(batch, episodes, length)
            local = cached["local"]
            if stage == "probe":
                with torch.no_grad():
                    output = self.reference.actor.decode_actions(local, local, alive, context)
                terms = self.probe_objective(output, batch, 0 if validation else update, weight_override=1.)
            else:
                output = self.learner.actor.decode_actions(local, local, alive, context)
                terms = {"distillation_loss": action_kl_term(output.values, cached["logits"], batch)}
                if self.policy_aux_weight:
                    extra = self.student_relation(output, batch, 0 if validation else update,
                                                    weight_override=self.policy_aux_weight)
                    terms.update(extra)
        total, active, metrics = None, 0., {}
        for name, term in terms.items():
            count = backend.sum(term.count.to(device=backend.device, dtype=torch.float32))
            if term.weight and count:
                contribution = term.weight * term.total / count
                total = contribution if total is None else total + contribution
                active += float(count)
            metrics[name] = float(backend.sum(term.total) / count.clamp_min(1))
            metrics[name + "_count"] = float(count)
        return total, active, metrics

    def _validation_sources(self):
        sources = [source for key, episode in self.offline_validation.items()
                   for source in partition_episode(key, episode, self.sequence_length, burn_in=0)]
        # Fixed independent sample for before/after measurements, never training.
        generator = torch.Generator().manual_seed(derive_seed(self.seed or 0, 122))
        indices = torch.randperm(len(sources), generator=generator)[:self.offline_batch_sequences].tolist()
        return [sources[index] for index in indices]

    @torch.no_grad()
    def _validate_offline(self, stage):
        backend = self.learner.backend
        sources = self._validation_sources()[backend.rank::backend.world_size]
        _, _, metrics = self._offline_loss(stage, self.offline_validation, sources, 0, validation=True)
        return metrics

    def run_offline(self, *, max_updates=None, checkpoint_interval=None):
        """Run/resume offline stages; optional bounds permit interactive inspection."""
        if not self._at_boundary:
            raise RuntimeError("Cannot continue a partially failed update")
        for name, value, minimum in (("max_updates", max_updates, 0), ("checkpoint_interval", checkpoint_interval, 1)):
            if value is not None and (type(value) is not int or value < minimum):
                raise ValueError(f"{name} has an invalid update count")
        if self.phase == "online" or max_updates == 0:
            return self.offline_metrics
        if not self.offline_train:
            self._at_boundary = False
            self.offline_train = self._collect_teacher(self.offline_train_episodes, 123, "offline_train")
            self.offline_validation = self._collect_teacher(self.offline_validation_episodes, 124, "offline_validation")
            self._at_boundary = True
        sources = [source for key, episode in self.offline_train.items()
                   for source in partition_episode(key, episode, self.sequence_length, burn_in=0)]
        completed = 0
        backend = self.learner.backend
        while self.phase != "online":
            stage = self.phase
            if self.offline_counters[stage] >= self.stage_steps[stage]:
                if stage == "probe" and self.probe_objective is not None:
                    self.probe_objective.requires_grad_(False).eval()
                index = self._stages.index(stage) + 1
                self.phase = self._stages[index] if index < len(self._stages) else "online"
                continue
            if max_updates is not None and completed >= max_updates:
                break
            if stage not in self.offline_metrics:
                self.offline_metrics[stage] = {"validation_initial": self._validate_offline(stage)}
            self._at_boundary = False
            indices = torch.randint(len(sources), (self.offline_batch_sequences,), generator=self.offline_rng).tolist()
            selected = [sources[index] for index in indices][backend.rank::backend.world_size]
            modules = [self.learner.actor]
            if self.probe_objective is not None:
                modules.append(self.probe_objective)
            if self.student_relation is not None:
                modules.append(self.student_relation)
            parameters = self.stage_parameters[stage]
            optimizer = self.stage_optimizers[stage]
            with _stage_trainability(modules, parameters):
                optimizer.zero_grad(set_to_none=True)
                loss, active, metrics = self._offline_loss(stage, self.offline_train, selected,
                                                           self.offline_counters[stage])
                if active:
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Nonfinite offline {stage} loss")
                    loss.backward()
                    backend.reduce_gradients(parameters)
                    torch.nn.utils.clip_grad_norm_(parameters, self.learner.max_grad_norm)
                    optimizer.step()
            self.offline_counters[stage] += 1
            completed += 1
            self.offline_metrics[stage]["last_training"] = metrics
            if self.offline_counters[stage] == self.stage_steps[stage]:
                self.offline_metrics[stage]["validation_final"] = self._validate_offline(stage)
            self._at_boundary = True
            total_updates = sum(self.offline_counters.values())
            if self.workdir is not None:
                if backend.rank == 0:
                    self.workdir.mkdir(parents=True, exist_ok=True)
                    with (self.workdir / "offline_events.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({"stage": stage, "update": self.offline_counters[stage],
                                                 "training": metrics}, allow_nan=False) + "\n")
                if checkpoint_interval is not None and total_updates % checkpoint_interval == 0:
                    self.save_checkpoint(self.workdir / "checkpoints" / f"offline_{total_updates}.pt")
        return self.offline_metrics

    def train_iteration(self, n_episodes=8):
        self.run_offline()
        return super().train_iteration(n_episodes)

    def _spec(self):
        result = super()._spec()
        result["offline"] = {"enabled": self.offline_enabled, "train_episodes": self.offline_train_episodes,
                             "validation_episodes": self.offline_validation_episodes,
                             "batch_sequences": self.offline_batch_sequences, "stage_steps": self.stage_steps,
                             "replacement": self.replacement, "boundaries": self.replacement_boundaries,
                             "codec_aux_weight": self.codec_aux_weight, "policy_aux_weight": self.policy_aux_weight}
        return result

    def _checkpoint_extra(self):
        return {"phase": self.phase, "offline_counters": self.offline_counters,
                "teacher": self.reference.actor.state_dict(), "teacher_critic": self.teacher_critic.state_dict(),
                "offline_rng": self.offline_rng.get_state(), "offline_metrics": self.offline_metrics,
                "probe": None if self.probe_objective is None else self.probe_objective.state_dict(),
                "optimizers": {name: None if optimizer is None else optimizer.state_dict()
                               for name, optimizer in self.stage_optimizers.items()},
                "datasets": {name: {key: episode.state_dict() for key, episode in episodes.items()}
                             for name, episodes in (("train", self.offline_train), ("validation", self.offline_validation))}}

    def _load_checkpoint_extra(self, state):
        self.reference.actor.load_state_dict(state["teacher"])
        self.teacher_critic.load_state_dict(state["teacher_critic"])
        self.reference.clear()
        self.phase = state["phase"]
        self.offline_counters = dict(state["offline_counters"])
        self.offline_rng.set_state(state["offline_rng"])
        self.offline_metrics = state["offline_metrics"]
        if (self.probe_objective is None) != (state["probe"] is None):
            raise ValueError("Checkpoint probe composition differs")
        if self.probe_objective is not None:
            self.probe_objective.load_state_dict(state["probe"])
            if self.offline_counters["probe"] >= self.stage_steps["probe"]:
                self.probe_objective.requires_grad_(False).eval()
        for name, optimizer in self.stage_optimizers.items():
            if (optimizer is None) != (state["optimizers"][name] is None):
                raise ValueError("Checkpoint offline optimizers differ")
            if optimizer is not None:
                optimizer.load_state_dict(state["optimizers"][name])
        self.offline_train = {key: Episode.from_state_dict(episode) for key, episode in state["datasets"]["train"].items()}
        self.offline_validation = {key: Episode.from_state_dict(episode) for key, episode in state["datasets"]["validation"].items()}
