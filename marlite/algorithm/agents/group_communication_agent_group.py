"""Shared finite-rate communication with parallel MAPPO and QMIX adapters."""

import numpy as np
import torch
from torch import nn
from marlite.algorithm.agents.agent_group import AgentGroup
from marlite.algorithm.group_builder import GroupBuilderConfig
from marlite.algorithm.model.group_communication import (
    ScalarMessageCodec,
    GroupSetFusion,
    RelationalNeuralProcess,
)
from marlite.util.action_distribution import masked_categorical


class GroupCommunicationAgentGroup(AgentGroup):
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
        if message_mode not in ("raw", "quantized"):
            raise ValueError("message_mode must be raw or quantized")
        self.message_mode = message_mode
        self.mask_observation_channels = tuple(mask_observation_channels)
        from marlite.util.communication_training import StateThresholdQueries

        self.relation_targets = (
            StateThresholdQueries(**relation_targets) if relation_targets else None
        )
        self.action_dim = action_dim
        self.auxiliary_weight = auxiliary_weight
        self.group_builder = GroupBuilderConfig(**group_builder).get_group_builder()
        self.features = nn.Sequential(nn.Linear(observation_dim, hidden_dim), nn.GELU())
        self.history = nn.GRUCell(hidden_dim, hidden_dim)
        self.codec = ScalarMessageCodec(hidden_dim, message_dim, bits)
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

            teacher = deepcopy(self)
            teacher.load_state_dict(
                torch.load(teacher_checkpoint, map_location="cpu", weights_only=True)
            )
            teacher.message_mode = "raw"
            teacher.distillation_steps = 0
            self.teacher = teacher.requires_grad_(False).eval()
        self.snapshot_step = 0

    def train(self, mode=True):
        super().train(mode)
        if self.teacher is not None:
            self.teacher.eval()
        return self

    def advance_training_step(self):
        self.distillation_updates.add_(1)

    def reset(self):
        self.group_builder.reset()
        self.snapshot_step = 0
        return self

    def forward(
        self,
        observations,
        traj_padding_mask,
        alive_mask,
        group_indices=None,
        message_override=None,
        replace_mask=None,
    ):
        if group_indices is None:
            raise ValueError(
                "Communication forward requires the recorded group snapshot"
            )
        b, n, t = observations.shape[:3]
        if traj_padding_mask.shape != (b, n, t) or alive_mask.shape != (b, n):
            raise ValueError("Observation and mask axes must be B,N,T")
        if self.mask_observation_channels:
            if observations.ndim != 6:
                raise ValueError("Channel masking expects B,N,T,H,W,C observations")
            observations = observations.clone()
            observations[..., list(self.mask_observation_channels)] = 0
        x = self.features(observations.reshape(b * n, t, -1))
        h = x.new_zeros(b * n, x.shape[-1])
        for step in range(t):
            candidate = self.history(x[:, step], h)
            h = torch.where(~traj_padding_mask[:, :, step].reshape(-1, 1), candidate, h)
        h = h.reshape(b, n, -1)
        decoded, codes = self.codec(h)
        codes = codes.masked_fill(~alive_mask.unsqueeze(-1), 0)
        tokens = h if self.message_mode == "raw" else decoded
        if message_override is not None:
            if (
                message_override.shape != h.shape
                or replace_mask.shape != alive_mask.shape
            ):
                raise ValueError("Replacement messages and mask must match sender axes")
            tokens = torch.where(replace_mask.unsqueeze(-1), message_override, tokens)
        memory, valid = self.fusion(tokens, group_indices, alive_mask)
        context = memory.mean(-2)
        policy_context = self.np.policy_context(context) if self.np else context
        indices = torch.where(alive_mask, group_indices, 0)
        rows = torch.arange(b, device=h.device)[:, None]
        agent_memory = memory[rows, indices]
        if self.np:
            agent_memory = agent_memory + (policy_context - context)[
                rows, indices
            ].unsqueeze(-2)
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
            loss = self.relation_loss(ret, batch)
            if loss is not None:
                ret["auxiliary_loss"] = self.auxiliary_weight * loss
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
        if self.relation_targets is not None:
            batch = dict(
                batch, **self.relation_targets(batch, ret["group_valid"], device)
            )
        if "relation_queries" not in batch:
            return None
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
        groups = (
            self.group_builder(torch.as_tensor(state, device=self.device).unsqueeze(0))
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
        if self.output_key == "action_logits":
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
        self.snapshot_step += 1
        return result


class GroupCommunicationMAPPOAgentGroup(GroupCommunicationAgentGroup):
    output_key = "action_logits"


class GroupCommunicationQMIXAgentGroup(GroupCommunicationAgentGroup):
    output_key = "q_val"
