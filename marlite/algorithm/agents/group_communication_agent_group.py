"""Experimental shared finite-rate communication with parallel MAPPO and QMIX adapters."""

import numpy as np
import torch
from torch import nn
from marlite.algorithm.agents.agent_group import AgentGroup
from marlite.algorithm.group_builder import GroupBuilderConfig
from marlite.algorithm.model.cp04_baselines import (
    CP04Baseline,
    CP04_BASELINES,
    gaussian_product,
    group_members,
)
from marlite.algorithm.model.group_communication import (
    ScalarMessageCodec,
    GaussianMessageCodec,
    StructuredPursuitCodec,
    GroupSetFusion,
    RelationalNeuralProcess,
)
from marlite.util.action_distribution import masked_categorical


class GroupCommunicationAgentGroup(AgentGroup):
    """Shared local encoder and group-conditioned action/Q reader.

    Observations use (batch, agent, time, ...); groups and alive masks use
    (batch, agent). Group slots are indexed by recorded group IDs, not by
    their order of appearance. Only auxiliary supervision may read labels.
    """

    output_key = None

    def __init__(
        self,
        agent_model_dict,
        observation_dim,
        action_dim,
        group_builder,
        hidden_dim=128,
        message_dim=32,
        bits=4,
        heads=4,
        slots=8,
        use_np=False,
        query_dim=4,
        latent_dim=32,
        hyper_rank=0,
        auxiliary_weight=0.1,
        message_mode="quantized",
        mask_observation_channels=(),
        relation_targets=None,
        teacher_checkpoint=None,
        distillation_steps=0,
        distillation_weight=0.1,
        feature_extractor=None,
        teacher_config=None,
        message_map_size=None,
        encoder_lr_scale=1.0,
        experimental_message_codec="learned",
        experimental_cp04_baseline=None,
    ):
        super().__init__()
        if (
            not agent_model_dict
            or observation_dim <= 0
            or action_dim <= 0
            or auxiliary_weight < 0
        ):
            raise ValueError("Invalid communication agent dimensions or weight")
        if len(set(agent_model_dict.values())) != 1:
            raise ValueError("Group communication currently requires a shared model")
        self.agent_model_dict = dict(agent_model_dict)
        if message_mode not in ("raw", "quantized", "local"):
            raise ValueError("message_mode must be raw, quantized or local")
        self.message_mode = message_mode
        if message_map_size is not None and (
            type(message_map_size) is not int or not 1 <= message_map_size <= 256
        ):
            raise ValueError("Version-1 coordinates require map size in [1,256]")
        self.message_map_size = message_map_size
        self.mask_observation_channels = tuple(mask_observation_channels)
        from marlite.util.communication_training import StateThresholdQueries

        self.relation_targets = None
        if relation_targets:
            target_config = dict(relation_targets)
            target_type = target_config.pop("type", "StateThresholdQueries")
            if target_type == "ComplementaryAttackQueries":
                from marlite.util.complementary_targets import (
                    ComplementaryAttackQueries,
                )

                if query_dim != ComplementaryAttackQueries.query_dim:
                    raise ValueError(
                        "Complementary attack queries require query_dim=14"
                    )
                self.relation_targets = ComplementaryAttackQueries(**target_config)
            elif target_type == "StateThresholdQueries":
                self.relation_targets = StateThresholdQueries(**target_config)
            else:
                raise ValueError(f"Unknown relation target constructor: {target_type}")
        if not 0 < encoder_lr_scale <= 1:
            raise ValueError("encoder_lr_scale must lie in (0,1]")
        self.encoder_lr_scale = encoder_lr_scale
        self.action_dim = action_dim
        self.auxiliary_weight = auxiliary_weight
        self.group_builder = GroupBuilderConfig(**group_builder).get_group_builder()
        self.flatten_observation = feature_extractor is None
        if feature_extractor is None:
            self.features = nn.Sequential(
                nn.Linear(observation_dim, hidden_dim),
                nn.GELU(),
            )
        else:
            from marlite.algorithm.model import ModelConfig

            self.features = ModelConfig(**feature_extractor).get_model()
        for module in self.features.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                raise ValueError(
                    "Local message encoders require per-sample normalization, not BatchNorm"
                )
            if (
                isinstance(module, (nn.MultiheadAttention, nn.RNNBase))
                and module.dropout > 0
            ):
                raise ValueError("Local attention/recurrent dropout must be zero")
            if isinstance(module, nn.modules.dropout._DropoutNd) and module.p > 0:
                raise ValueError(
                    "Message encoders must be deterministic for PPO probability replay"
                )
        self.history = nn.GRUCell(hidden_dim, hidden_dim)
        if experimental_cp04_baseline is not None and (
            experimental_cp04_baseline not in CP04_BASELINES
            or experimental_message_codec != "learned"
            or message_mode != "quantized"
            or (message_dim, bits) != (32, 4)
            or message_map_size is None
            or use_np or hyper_rank or teacher_checkpoint is not None
        ):
            raise ValueError("CP04 adaptations require plain 128-bit quantized pursuit agents")
        self.cp04_baseline = (
            None if experimental_cp04_baseline is None else
            CP04Baseline(experimental_cp04_baseline, hidden_dim, action_dim, message_map_size)
        )
        if experimental_message_codec == "learned":
            codec = GaussianMessageCodec if experimental_cp04_baseline == "ss_gigc" else ScalarMessageCodec
            self.codec = codec(hidden_dim, message_dim, bits)
        elif experimental_message_codec == "structured_pursuit":
            if message_mode != "quantized" or (message_dim, bits) != (32, 4):
                raise ValueError("Structured pursuit requires 128-bit quantized messages")
            self.codec = StructuredPursuitCodec(hidden_dim, message_map_size)
        else:
            raise ValueError("Unknown experimental message codec")
        self.fusion = GroupSetFusion(hidden_dim, heads, slots)
        self.reader = nn.MultiheadAttention(
            hidden_dim, heads, batch_first=True, dropout=0
        )
        self.local_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU()
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.np = (
            RelationalNeuralProcess(hidden_dim, query_dim, latent_dim)
            if use_np
            else None
        )
        self.relation_head = nn.Sequential(
            nn.Linear(hidden_dim + query_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        if hyper_rank < 0:
            raise ValueError("hyper_rank cannot be negative")
        self.hyper_rank = hyper_rank
        self.hyper = (
            nn.Linear(
                hidden_dim,
                action_dim * hyper_rank + hyper_rank * hidden_dim + action_dim,
            )
            if hyper_rank
            else None
        )
        if distillation_steps < 0 or distillation_weight < 0:
            raise ValueError("Distillation steps and weight must be nonnegative")
        if teacher_config is not None and teacher_checkpoint is None:
            raise ValueError("teacher_config requires a trained teacher checkpoint")
        if teacher_checkpoint is None and distillation_steps:
            raise ValueError(
                "Online distillation requires a trained raw teacher checkpoint"
            )
        self.register_buffer("distillation_updates", torch.zeros((), dtype=torch.long))
        self.distillation_steps = distillation_steps
        self.distillation_weight = distillation_weight
        self.teacher = None
        if teacher_checkpoint is not None:
            from copy import deepcopy

            if teacher_config is None:
                teacher = deepcopy(self)
            else:
                from marlite.algorithm.agents.agent_group_config import AgentGroupConfig

                if teacher_config.get("teacher_checkpoint") is not None:
                    raise ValueError(
                        "A teacher cannot recursively contain another teacher"
                    )
                teacher = AgentGroupConfig(**teacher_config).get_agent_group()
                if teacher.message_mode != "raw":
                    raise ValueError(
                        "teacher_config must describe a raw-message teacher"
                    )
                if (
                    teacher.action_dim != self.action_dim
                    or teacher.output_key != self.output_key
                    or teacher.agent_model_dict != self.agent_model_dict
                ):
                    raise ValueError(
                        "Teacher algorithm and sender schema must match student"
                    )
                if teacher.mask_observation_channels != self.mask_observation_channels:
                    raise ValueError(
                        "Teacher and student observation permissions must match"
                    )
            teacher.load_state_dict(
                torch.load(teacher_checkpoint, map_location="cpu", weights_only=True)
            )
            teacher.message_mode = "raw"
            teacher.distillation_steps = 0
            self.teacher = teacher.requires_grad_(False).eval()
        self.snapshot_step = 0
        self._last_groups = None
        self.group_version = 0

    def train(self, mode=True):
        super().train(mode)
        if self.teacher is not None:
            self.teacher.eval()
        return self

    def optimizer_parameter_groups(self, base_lr):
        """Keep policy LR first so trainer/worker LR synchronization stays consistent."""
        encoder = [
            p
            for module in (self.features, self.history, self.codec)
            for p in module.parameters()
            if p.requires_grad
        ]
        encoder_ids = {id(p) for p in encoder}
        policy = [
            p for p in self.parameters() if p.requires_grad and id(p) not in encoder_ids
        ]
        return [
            {"params": policy, "lr": base_lr, "lr_scale": 1.0},
            {
                "params": encoder,
                "lr": base_lr * self.encoder_lr_scale,
                "lr_scale": self.encoder_lr_scale,
            },
        ]

    def advance_training_step(self):
        self.distillation_updates.add_(1)

    def reset(self):
        self.group_builder.reset()
        self.snapshot_step = 0
        self._last_groups = None
        self.group_version = 0
        return self

    def encode_local(self, observations, traj_padding_mask):
        """Encode each sender independently; no group/state/partner input is accepted."""
        b, n, t = observations.shape[:3]
        if traj_padding_mask.shape != (b, n, t):
            raise ValueError("History padding must have shape B,N,T")
        if self.mask_observation_channels:
            if observations.ndim != 6:
                raise ValueError("Channel masking expects B,N,T,H,W,C observations")
            observations = observations.clone()
            observations[..., list(self.mask_observation_channels)] = 0
        frames = observations.reshape(b * n * t, *observations.shape[3:])
        x = self.features(frames.flatten(1) if self.flatten_observation else frames)
        x = x.reshape(b * n, t, -1)
        if x.shape[-1] != self.history.input_size:
            raise ValueError(
                "Feature extractor must output hidden_dim features per frame"
            )
        h = x.new_zeros(b * n, x.shape[-1])
        for step in range(t):
            candidate = self.history(x[:, step], h)
            h = torch.where(~traj_padding_mask[:, :, step].reshape(-1, 1), candidate, h)
        h = h.reshape(b, n, -1)
        if self.message_mode == "quantized":
            source = observations[:, :, -1] if isinstance(self.codec, StructuredPursuitCodec) else h
            decoded, codes = self.codec(source)
        else:
            decoded = h
            codes = torch.empty(b, n, 0, dtype=torch.uint8, device=h.device)
        return h, decoded, codes

    def forward(
        self,
        observations,
        traj_padding_mask,
        alive_mask,
        group_indices=None,
        message_override=None,
        replace_mask=None,
        detach_messages=False,
    ):
        if group_indices is None:
            raise ValueError(
                "Communication forward requires the recorded group snapshot"
            )
        b, n, t = observations.shape[:3]
        if traj_padding_mask.shape != (b, n, t) or alive_mask.shape != (b, n):
            raise ValueError("Observation and mask axes must be B,N,T")
        h, decoded, codes = self.encode_local(observations, traj_padding_mask)
        if detach_messages:
            h, decoded = h.detach(), decoded.detach()
        codes = codes.masked_fill(~alive_mask.unsqueeze(-1), 0)
        tokens = h if self.message_mode in ("raw", "local") else decoded
        if self.message_mode == "local":
            group_indices = torch.arange(n, device=h.device).expand(b, -1)
            codes = codes[..., :0]
        if message_override is not None:
            if self.cp04_baseline is not None:
                raise ValueError("Paper adaptations do not use CP04's message replacement curriculum")
            if (
                message_override.shape != h.shape
                or replace_mask.shape != alive_mask.shape
            ):
                raise ValueError("Replacement messages and mask must match sender axes")
            tokens = torch.where(replace_mask.unsqueeze(-1), message_override, tokens)
        baseline = self.cp04_baseline
        baseline_context = None
        if baseline is not None and baseline.method == "ss_gigc":
            baseline_context, valid = gaussian_product(tokens, group_members(group_indices, alive_mask))
            mean, logvar = baseline_context.chunk(2, -1)
            memory = self.codec.decoder(
                torch.cat((mean, torch.zeros_like(logvar)), -1)
            ).unsqueeze(-2)
        else:
            memory, valid = self.fusion(
                tokens, group_indices, alive_mask, singleton_groups=self.message_mode == "local"
            )
            if baseline is not None:
                memory, baseline_context = baseline.context(memory)
        context = memory.mean(-2)
        policy_context = self.np.policy_context(context) if self.np else context
        indices = torch.where(alive_mask, group_indices, 0)
        rows = torch.arange(b, device=h.device)[:, None]
        agent_memory = memory[rows, indices]
        if self.np:
            agent_memory = agent_memory + (policy_context - context)[
                rows, indices
            ].unsqueeze(-2)
        if baseline is not None and baseline.method == "masia":
            read = baseline.read(h, agent_memory)
        else:
            read, _ = self.reader(
                h.reshape(b * n, 1, -1),
                agent_memory.flatten(0, 1),
                agent_memory.flatten(0, 1),
                need_weights=False,
            )
        features = self.local_head(torch.cat([h, read.reshape(b, n, -1)], -1))
        values = self.action_head(features)
        if self.hyper:
            cond = policy_context[rows, indices]
            parameters = self.hyper(cond)
            rank, hidden = self.hyper_rank, h.shape[-1]
            left, right, bias = parameters.split(
                [self.action_dim * rank, rank * hidden, self.action_dim], -1
            )
            residual = right.reshape(b, n, rank, hidden) @ features.unsqueeze(-1)
            values = (
                values
                + (left.reshape(b, n, self.action_dim, rank) @ residual).squeeze(-1)
                + bias
            )
        values = values.masked_fill(~alive_mask.unsqueeze(-1), 0)
        return {
            self.output_key: values,
            "message_codes": codes,
            "group_indices": group_indices,
            "raw_tokens": h,
            "decoded_tokens": decoded,
            "group_tokens": memory,
            "group_valid": valid,
            "relation_context": context,
            **({"baseline_context": baseline_context} if baseline is not None else {}),
        }

    def training_forward(self, observations, padding, alive, batch, next_state=False):
        key = "next_group_indices" if next_state else "group_indices"
        if key not in batch:
            raise ValueError(
                f"Missing {key}; select the group communication rollout profile"
            )
        groups = batch[key][:, -1].to(device=observations.device, dtype=torch.long)
        ret = self(observations, padding, alive, groups)
        if not next_state:
            if self.cp04_baseline is not None:
                loss = self.cp04_baseline.loss(ret, batch, alive, groups)
            elif self.auxiliary_weight:
                loss = self.relation_loss(ret, batch)
            else:
                # Keep the same query draws while excluding relation gradients.
                with torch.no_grad():
                    self.relation_loss(ret, batch)
                loss = None
            if loss is not None:
                ret["auxiliary_loss"] = self.auxiliary_weight * loss
                if self.cp04_baseline is None:
                    ret["relation_auxiliary_loss"] = ret["auxiliary_loss"]
        if not next_state and self.teacher is not None and self.distillation_steps:
            from marlite.util.communication_curriculum import action_distillation

            weight = self.distillation_weight * max(
                0.0, 1.0 - self.distillation_updates.item() / self.distillation_steps
            )
            if weight:
                with torch.no_grad():
                    target = self.teacher(observations, padding, alive, groups)[
                        self.output_key
                    ]
                available = batch.get("avail_actions")
                available = (
                    available[:, -1].to(observations.device).bool()
                    if isinstance(available, torch.Tensor)
                    else None
                )
                loss = action_distillation(
                    ret[self.output_key], target, alive, self.output_key, available
                )
                ret["auxiliary_loss"] = ret.get("auxiliary_loss", 0.0) + weight * loss
        return ret

    def relation_loss(self, ret, batch):
        device = ret["relation_context"].device
        fields = ("relation_queries", "relation_labels", "relation_mask")
        if not any(field in batch for field in fields):
            if self.relation_targets is None:
                return None
            batch = self.relation_targets(batch, ret["group_valid"], device)
        if not all(field in batch for field in fields):
            raise ValueError("Supply relation queries, labels and mask together")
        queries = batch["relation_queries"].to(device=device, dtype=torch.float32)
        labels = batch["relation_labels"].to(device=device, dtype=torch.float32)
        mask = batch["relation_mask"].to(device).bool()
        if (
            queries.shape[:2] != ret["group_valid"].shape
            or queries.shape[:-1] != labels.shape
            or mask.shape != labels.shape
        ):
            raise ValueError(
                "Relation targets must match the snapshot's group/query axes"
            )
        mask = mask & ret["group_valid"].unsqueeze(-1)
        if (
            not torch.isfinite(queries).all()
            or not torch.isfinite(labels[mask]).all()
            or torch.any((labels[mask] < 0) | (labels[mask] > 1))
        ):
            raise ValueError("Relation queries must be finite and labels in [0,1]")
        if self.np:
            return self.np.loss(ret["relation_context"], queries, labels, mask)
        context = ret["relation_context"].unsqueeze(-2).expand(*queries.shape[:-1], -1)
        logits = self.relation_head(torch.cat([context, queries], -1)).squeeze(-1)
        safe = torch.where(mask, labels, torch.zeros_like(labels))
        return (
            nn.functional.binary_cross_entropy_with_logits(
                logits, safe, reduction="none"
            )
            * mask
        ).sum() / mask.sum().clamp_min(1)

    @torch.no_grad()
    def act(
        self,
        observations,
        state,
        avail_actions,
        traj_padding_mask,
        alive_agents,
        epsilon=0.0,
        *,
        greedy_evaluation=False,
    ):
        names = list(self.agent_model_dict)
        obs = torch.as_tensor(
            np.stack([observations[a] for a in names]),
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        padding = torch.as_tensor(
            traj_padding_mask, device=self.device, dtype=torch.bool
        )[None, None].expand(1, len(names), -1)
        alive = torch.tensor([[a in alive_agents for a in names]], device=self.device)
        if self.message_mode == "local":
            groups = torch.arange(len(names), device=self.device).unsqueeze(0)
        else:
            groups = (
                self.group_builder(
                    torch.as_tensor(state, device=self.device).unsqueeze(0)
                )
                .to(self.device)
                .long()
            )
        groups = groups.masked_fill(~alive, -1)
        ret = self(obs, padding, alive, groups)
        values = ret[self.output_key][0]
        masks = None
        if isinstance(next(iter(avail_actions.values())), np.ndarray):
            masks = torch.as_tensor(
                np.stack([avail_actions[a] for a in names]),
                device=self.device,
                dtype=torch.bool,
            )
        dist = masked_categorical(values, masks, active_mask=alive[0])
        if greedy_evaluation:
            actions = dist.logits.argmax(-1)
        elif self.output_key == "action_logits":
            actions = dist.sample()
        else:
            if not 0 <= epsilon <= 1:
                raise ValueError("epsilon must lie in [0,1]")
            greedy = (
                values.masked_fill(~masks, -torch.inf).argmax(-1)
                if masks is not None
                else values.argmax(-1)
            )
            random = masked_categorical(
                torch.zeros_like(values), masks, active_mask=alive[0]
            ).sample()
            actions = torch.where(
                torch.rand(len(names), device=self.device) < epsilon, random, greedy
            )
        actions = actions.masked_fill(~alive[0], 0)
        all_actions = dict(zip(names, actions.cpu().tolist()))
        result = {
            "actions": {a: all_actions[a] for a in alive_agents},
            "all_actions": all_actions,
            "all_group_indices": dict(zip(names, groups[0].cpu().tolist())),
            "message_codes": dict(zip(names, ret["message_codes"][0].cpu().numpy())),
            "snapshot_steps": {a: self.snapshot_step for a in names},
        }
        if self.output_key == "action_logits":
            result["all_log_probs"] = dict(
                zip(
                    names,
                    dist.log_prob(actions).masked_fill(~alive[0], 0).cpu().tolist(),
                )
            )
        # Serialize the actual payload for byte accounting; transport remains synchronous.
        from marlite.util.communication_protocol import MessagePacket, snapshot_packets

        current_groups = groups[0].cpu().tolist()
        if self._last_groups is not None and current_groups != self._last_groups:
            self.group_version += 1
        self._last_groups = current_groups
        packets = []
        sizes = dict.fromkeys(names, 0)
        if self.message_mode != "local":
            for sender, name in enumerate(names):
                if name not in alive_agents:
                    continue
                if self.message_mode == "raw":
                    payload = (
                        ret["raw_tokens"][0, sender].float().cpu().numpy().tobytes()
                    )
                else:
                    payload = (
                        self.codec.pack(ret["message_codes"][0, sender].cpu())
                        .numpy()
                        .tobytes()
                    )
                x = y = 0
                if self.message_map_size is not None:
                    frame = observations[name][-1]
                    if frame.ndim != 3:
                        raise ValueError(
                            "Grid coordinate metadata requires H,W,C observations"
                        )
                    position = frame[frame.shape[0] // 2, frame.shape[1] // 2, -2:]
                    x, y = (
                        np.rint(position * self.message_map_size).astype(int).tolist()
                    )
                packet = MessagePacket(
                    sender, self.snapshot_step, self.group_version, x, y, payload
                )
                sizes[name] = len(packet.encode())
                packets.append(packet)
            for group in set(current_groups) - {-1}:
                senders = [
                    i
                    for i, g in enumerate(current_groups)
                    if g == group and names[i] in alive_agents
                ]
                snapshot_packets(
                    [p for p in packets if p.sender in senders],
                    self.snapshot_step,
                    self.group_version,
                    senders,
                )
        result["communication_bytes"] = sizes
        self.snapshot_step += 1
        return result


class GroupCommunicationMAPPOAgentGroup(GroupCommunicationAgentGroup):
    output_key = "action_logits"


class GroupCommunicationQMIXAgentGroup(GroupCommunicationAgentGroup):
    output_key = "q_val"
