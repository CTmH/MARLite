"""Offline message distillation and online MAPPO for group communication."""

import math
from copy import deepcopy
from pathlib import Path
from random import Random

import torch
from absl import logging

from marlite.algorithm.auxiliary import RelationPredictionHead
from marlite.rollout.attribute_spec import resolve_required_attrs
from marlite.trainer.mappo_trainer import MAPPOTrainer
from marlite.util.communication_curriculum import (
    action_distillation,
    sender_replacement_mask,
)
from marlite.util.communication_training import StateThresholdQueries
from marlite.util.complementary_targets import ComplementaryAttackQueries
from marlite.util.optimizer_config import OptimizerConfig
from marlite.util.randomness import derive_seed, preserve_rng_state, seed_everything
from marlite.util.serialization import get_state_dict, load_state_dict_into, serialize_to_buffer
from marlite.util.trajectory_dataset import TrajectoryDataset, trajectory_collate_fn


_AGENT_TYPE = "DistilledGroupCommunicationMAPPO"
_LABEL_PROVIDERS = {
    "ComplementaryAttackQueries": ComplementaryAttackQueries,
    "StateThresholdQueries": StateThresholdQueries,
}


def _optimizer(config, parameters):
    return OptimizerConfig(**config).get_optimizer(parameters)


def _read_actor(path, actor):
    state = torch.load(path, map_location="cpu", weights_only=True)
    load_state_dict_into(actor, state, strict=True)


class DistilledGroupCommunicationMAPPOTrainer(MAPPOTrainer):
    """Run the entire student training workflow from one MARLite config."""

    def __init__(
        self, teacher, teacher_agent_config, teacher_training,
        offline=None, online_distillation=None, auxiliary_head=None, **kwargs,
    ):
        if not isinstance(kwargs.get("train_device", "cpu"), str):
            raise ValueError("DistilledGroupCommunicationMAPPO requires a single learner device")
        super().__init__(**kwargs)
        if (
            self.agent_group_config.ag_type != _AGENT_TYPE
            or self.eval_agent_group.message_mode != "quantized"
        ):
            raise ValueError("The student requires a quantized DistilledGroupCommunicationMAPPO agent")

        self.teacher_config = teacher_agent_config
        self.teacher_options = teacher
        self.teacher_training = teacher_training
        self.teacher = None
        self.actor_updates = 0
        self._student_checkpoint_loaded = False
        self.offline_metrics = {}

        required = {"group_indices", "all_log_probs", "observations", "alive_mask"}
        for profile in (
            self.rolloutmanager_config.required_attrs, self.replaybuffer.required_attrs
        ):
            if required.difference(resolve_required_attrs(profile)):
                raise ValueError("Use a group_communication_mappo-compatible rollout and replay profile")

        self._configure_offline(**(offline or {}))
        self._configure_online_distillation(**(online_distillation or {}))
        self._configure_auxiliary(**(auxiliary_head or {}))

    def _configure_offline(
        self, enabled=True, train_episodes=32, validation_episodes=16, batch_size=32,
        codec_updates=600, policy_updates=200, replacement="all",
        replacement_steps=None, codec_optimizer=None, policy_optimizer=None,
    ):
        if type(enabled) is not bool:
            raise ValueError("offline.enabled must be a boolean")
        self.offline_enabled = enabled
        self.offline_train_episodes = train_episodes
        self.offline_validation_episodes = validation_episodes
        self.offline_batch_size = batch_size
        self.codec_updates = codec_updates
        self.policy_updates = policy_updates
        self.replacement = replacement
        self.replacement_boundaries = ()
        self.codec_optimizer_config = codec_optimizer or {"type": "Adam", "lr": 0.001}
        self.policy_optimizer_config = policy_optimizer or {"type": "Adam", "lr": 0.001}

        counts = (train_episodes, validation_episodes, batch_size, codec_updates, policy_updates)
        if any(type(count) is not int or count < 1 for count in counts):
            raise ValueError("Offline episode counts, batch size and update counts must be positive integers")
        if replacement == "progressive":
            if (
                not isinstance(replacement_steps, (list, tuple))
                or len(replacement_steps) != 3
                or any(type(step) is not int or step < 1 for step in replacement_steps)
                or sum(replacement_steps) != codec_updates
            ):
                raise ValueError("Provide three positive replacement_steps summing to codec_updates")
            self.replacement_boundaries = (
                replacement_steps[0], sum(replacement_steps[:2])
            )
        elif replacement != "all" or replacement_steps is not None:
            raise ValueError("Use replacement=all without replacement_steps, or replacement=progressive")

    def _configure_online_distillation(self, enabled=False, weight=0.1, decay_updates=1000):
        if (
            type(enabled) is not bool or not math.isfinite(weight) or weight < 0
            or type(decay_updates) is not int or decay_updates < 1
        ):
            raise ValueError("Invalid online distillation weight or decay schedule")
        self.online_distillation_weight = weight if enabled else 0.0
        self.online_distillation_updates = decay_updates

    def _configure_auxiliary(
        self, enabled=False, model=None, label_provider=None, probe_updates=600,
        codec_weight=1.0, policy_weight=1.0, online_weight=0.1, optimizer=None,
    ):
        if type(enabled) is not bool:
            raise ValueError("auxiliary_head.enabled must be a boolean")
        weights = (codec_weight, policy_weight, online_weight)
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("Auxiliary loss weights must be finite and nonnegative")
        self.probe_updates = probe_updates
        self.codec_aux_weight = codec_weight if enabled and self.offline_enabled else 0.0
        self.policy_aux_weight = policy_weight if enabled and self.offline_enabled else 0.0
        self.online_aux_weight = online_weight if enabled else 0.0
        self.auxiliary_model = model
        self.aux_optimizer_config = optimizer or {"type": "Adam", "lr": 0.001}
        self.label_provider = None
        self.student_auxiliary = None
        self.online_aux_optimizer = None

        if not (self.codec_aux_weight or self.policy_aux_weight or self.online_aux_weight):
            return
        if model is None or label_provider is None:
            raise ValueError("An active auxiliary head requires model and label_provider configs")
        if self.codec_aux_weight and (type(probe_updates) is not int or probe_updates < 1):
            raise ValueError("The codec auxiliary loss requires positive probe_updates")
        provider = dict(label_provider)
        provider_type = provider.pop("type")
        if provider_type not in _LABEL_PROVIDERS:
            raise ValueError(f"Unknown auxiliary label provider: {provider_type}")
        self.label_provider = _LABEL_PROVIDERS[provider_type](**provider)
        if self.policy_aux_weight or self.online_aux_weight:
            with preserve_rng_state():
                seed_everything(derive_seed(self.seed, 102))
                self.student_auxiliary = RelationPredictionHead(model).to(self.train_device)
        if self.online_aux_weight:
            self.online_aux_optimizer = _optimizer(
                self.aux_optimizer_config, self.student_auxiliary.parameters()
            )

    def _prepare_teacher(self):
        options = self.teacher_options
        if options["source"] == "checkpoint":
            teacher = self.teacher_config.get_agent_group()
            _read_actor(options["actor_checkpoint"], teacher)
        else:
            with preserve_rng_state():
                teacher_kwargs, train_args = self.teacher_training
                trainer = MAPPOTrainer(**teacher_kwargs)
                try:
                    if options.get("initial_actor_checkpoint"):
                        _read_actor(options["initial_actor_checkpoint"], trainer.eval_agent_group)
                    trainer.train(**train_args)
                    trainer.save_current_model("final")
                    teacher = trainer.eval_agent_group
                finally:
                    # The standard Trainer configures a process-global log handler.
                    logging.get_absl_handler().use_absl_log_file("training", self.logdir)
        if not self._student_checkpoint_loaded:
            state = get_state_dict(self.eval_agent_group)
            state.update({
                name: value for name, value in get_state_dict(teacher).items()
                if not name.startswith("codec.")
            })
            load_state_dict_into(self.eval_agent_group, state, strict=True)
        self.teacher = teacher.requires_grad_(False).eval().to(self.train_device)

    def _collect_offline(self, episodes, stream):
        config = deepcopy(self.rolloutmanager_config)
        config.n_episodes = episodes
        config.set_randomness(derive_seed(self.seed, stream), self.deterministic)
        weights = serialize_to_buffer({
            name: value.detach().cpu() for name, value in self.teacher.state_dict().items()
        })
        manager = config.create_manager(self.teacher_config, weights, self.env_config, epsilon=0.0)
        collected = manager.generate_episodes()
        if len(collected) != episodes or any(ep["episode_length"] < 1 for ep in collected):
            raise RuntimeError("Teacher did not return the configured number of complete episodes")
        windows = [
            (index, timestep)
            for index, episode in enumerate(collected)
            for timestep, alive in enumerate(episode["alive_mask"])
            if any(alive.values())
        ]
        if not windows:
            raise RuntimeError("Teacher collection contained no live-agent decision windows")
        fields = ["observations", "alive_mask", "group_indices", "avail_actions"]
        if self.codec_aux_weight or self.policy_aux_weight:
            fields.append("states")
        return TrajectoryDataset(
            windows, dict(enumerate(collected)), config.config["traj_len"], required_attrs=fields
        )

    def _sample_offline(self, dataset, rng):
        indices = rng.sample(range(len(dataset)), min(len(dataset), self.offline_batch_size))
        return trajectory_collate_fn([dataset[index] for index in indices])

    def _offline_inputs(self, batch):
        device = self.train_device
        observations = batch["observations"].to(device=device, dtype=torch.float32).transpose(1, 2)
        alive = batch["alive_mask"][:, -1].to(device=device, dtype=torch.bool)
        padding = batch["timestep_padding_mask"].to(device=device, dtype=torch.bool)
        padding = padding[:, None].expand(-1, alive.shape[1], -1)
        groups = batch["group_indices"][:, -1].to(device=device, dtype=torch.long)
        available = batch.get("avail_actions")
        available = (
            available[:, -1].to(device=device, dtype=torch.bool)
            if isinstance(available, torch.Tensor) else None
        )
        return (observations, padding, alive, groups), available

    def _offline_actor_loss(self, batch, stage, update, probe):
        inputs, available = self._offline_inputs(batch)
        alive, groups = inputs[2:]
        with torch.no_grad():
            target = self.teacher(*inputs)
        if stage == "codec":
            replacement_stage = sum(update >= end for end in self.replacement_boundaries)
            with preserve_rng_state():
                seed_everything(derive_seed(self.seed, 110, update))
                selected = sender_replacement_mask(
                    groups, alive, self.replacement, replacement_stage, update
                )
            _, decoded, _ = self.eval_agent_group.encode_local(*inputs[:2])
            # Frozen weights must still transmit the gradient to the replaced messages.
            predicted = self.teacher(*inputs, message_override=decoded, replace_mask=selected)
            head, weight = probe, self.codec_aux_weight
        else:
            predicted = self.eval_agent_group(*inputs, detach_messages=True)
            head, weight = self.student_auxiliary, self.policy_aux_weight
        loss = action_distillation(
            predicted["action_logits"], target["action_logits"], alive, available
        )
        if weight:
            with preserve_rng_state():
                seed_everything(derive_seed(self.seed, 107, update))
                loss = loss + weight * head.loss(predicted, batch, self.label_provider)
        return loss

    def _run_offline_stage(self, name, parameters, optimizer, steps, train, validation, loss_fn):
        rng = Random(derive_seed(self.seed, 105))
        validation_batch = self._sample_offline(validation, Random(derive_seed(self.seed, 106)))
        with torch.no_grad(), preserve_rng_state():
            seed_everything(derive_seed(self.seed, 108))
            initial = float(loss_fn(validation_batch, steps - 1))
        total = 0.0
        for update in range(steps):
            batch = self._sample_offline(train, rng)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(batch, update)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite offline {name} loss at update {update}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
            optimizer.step()
            total += float(loss.detach())

        with torch.no_grad(), preserve_rng_state():
            seed_everything(derive_seed(self.seed, 108))
            final = float(loss_fn(validation_batch, steps - 1))
        self.offline_metrics[name] = {
            "updates": steps, "loss": total / steps,
            "validation_initial": initial, "validation_final": final,
        }
        logging.info("Offline %s: %s", name, self.offline_metrics[name])

    def _train_offline(self):
        train = self._collect_offline(self.offline_train_episodes, 103)
        validation = self._collect_offline(self.offline_validation_episodes, 104)
        probe = None
        if self.codec_aux_weight:
            probe = RelationPredictionHead(self.auxiliary_model).to(self.train_device)
            parameters = list(probe.parameters())

            def probe_loss(batch, _update):
                inputs, _ = self._offline_inputs(batch)
                with torch.no_grad():
                    target = self.teacher(*inputs)
                return probe.loss(target, batch, self.label_provider)

            self._run_offline_stage(
                "probe", parameters, _optimizer(self.aux_optimizer_config, parameters),
                self.probe_updates, train, validation, probe_loss,
            )
            probe.requires_grad_(False).eval()

        student = self.eval_agent_group.to(self.train_device)
        student.requires_grad_(False)
        try:
            student.codec.requires_grad_(True)
            codec = list(student.codec.parameters())
            self._run_offline_stage(
                "codec", codec, _optimizer(self.codec_optimizer_config, codec),
                self.codec_updates, train, validation,
                lambda batch, update: self._offline_actor_loss(batch, "codec", update, probe),
            )
            student.codec.requires_grad_(False)

            modules = [student.fusion, student.reader, student.local_head, student.action_head]
            if self.policy_aux_weight:
                modules.append(self.student_auxiliary)
            parameters = [
                parameter for module in modules
                for parameter in module.requires_grad_(True).parameters()
            ]
            self._run_offline_stage(
                "policy", parameters, _optimizer(self.policy_optimizer_config, parameters),
                self.policy_updates, train, validation,
                lambda batch, update: self._offline_actor_loss(batch, "policy", update, probe),
            )
        finally:
            student.zero_grad(set_to_none=True)
            student.requires_grad_(True).to("cpu")

    def _actor_optimizers(self):
        optimizers = super()._actor_optimizers()
        return optimizers if self.online_aux_optimizer is None else (*optimizers, self.online_aux_optimizer)

    def _on_actor_update(self):
        self.actor_updates += 1

    def _minibatch_actor_objectives(self, actor_result, batch, observations, padding, alive, available):
        objectives = {}
        weight = self.online_distillation_weight * max(
            0.0, 1.0 - self.actor_updates / self.online_distillation_updates
        )
        if weight:
            groups = actor_result["group_indices"]
            with torch.no_grad():
                target = self.teacher(observations, padding, alive, groups)["action_logits"]
            objectives["online_distillation_loss"] = weight * action_distillation(
                actor_result["action_logits"], target, alive, available
            )
        if self.online_aux_weight:
            with preserve_rng_state():
                seed_everything(derive_seed(self.seed, 109, self.actor_updates))
                auxiliary = self.student_auxiliary.loss(actor_result, batch, self.label_provider)
            objectives["online_auxiliary_loss"] = self.online_aux_weight * auxiliary
        return objectives

    def learn(self, *args, **kwargs):
        result = super().learn(*args, **kwargs)
        logging.info("C04 online after %d actor updates: %s", self.actor_updates, result)
        return result

    def load_checkpoint(self, checkpoint):
        if self.offline_enabled:
            raise ValueError("Set offline.enabled=false to start online from a student checkpoint")
        super().load_checkpoint(checkpoint)
        if self.online_aux_weight:
            path = Path(self.checkpointdir) / checkpoint / "auxiliary.pth"
            _read_actor(path, self.student_auxiliary)
        self._student_checkpoint_loaded = True
        return self

    def save_current_model(self, checkpoint):
        super().save_current_model(checkpoint)
        if self.student_auxiliary is not None:
            state = {key: value.detach().cpu() for key, value in self.student_auxiliary.state_dict().items()}
            torch.save(state, Path(self.checkpointdir) / checkpoint / "auxiliary.pth")
        return self

    def train(self, **kwargs):
        with preserve_rng_state():
            seed_everything(derive_seed(self.seed, 101))
            if not self._student_checkpoint_loaded or self.online_distillation_weight:
                self._prepare_teacher()
            if self.offline_enabled:
                self._train_offline()
        if not self.online_distillation_weight:
            self.teacher = None
        result = super().train(**kwargs)
        self.save_current_model("final")
        return result
