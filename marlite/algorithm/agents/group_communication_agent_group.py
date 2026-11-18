"""Group communication policy shared by the raw teacher and quantized student."""

import numpy as np
import torch
from torch import nn
from marlite.algorithm.agents.agent_group import AgentGroup
from marlite.algorithm.group_builder import GroupBuilderConfig
from marlite.algorithm.model import ModelConfig
from marlite.algorithm.model.group_communication import GroupSetFusion, ScalarMessageCodec
from marlite.util.action_distribution import masked_categorical


class DistilledGroupCommunicationMAPPOAgentGroup(AgentGroup):
    """Shared local encoder and group-conditioned action reader.

    Observations use (batch, agent, time, ...); groups and alive masks use
    (batch, agent). Group slots are indexed by recorded group IDs, not by
    their order of appearance. Only auxiliary supervision may read labels.
    """

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
        message_mode="quantized",
        mask_observation_channels=(),
        feature_extractor=None,
        message_map_size=None,
        encoder_lr_scale=1.0,
        codec_encoder=None,
        codec_decoder=None,
        history_bias=True,
        history_encoder=None,
        inducing_points=16,
        reader_heads=None,
        local_head=None,
        action_head=None,
    ):
        super().__init__()
        if (
            not agent_model_dict
            or observation_dim <= 0
            or action_dim <= 0
        ):
            raise ValueError("Invalid communication agent dimensions or weight")
        if len(set(agent_model_dict.values())) != 1:
            raise ValueError("Group communication currently requires a shared model")
        self.agent_model_dict = dict(agent_model_dict)
        if message_mode not in ("raw", "quantized"):
            raise ValueError("message_mode must be raw or quantized")
        self.message_mode = message_mode
        if message_map_size is not None and (
            type(message_map_size) is not int or not 1 <= message_map_size <= 256
        ):
            raise ValueError("Version-1 coordinates require map size in [1,256]")
        self.message_map_size = message_map_size
        self.mask_observation_channels = tuple(mask_observation_channels)
        if not 0 < encoder_lr_scale <= 1:
            raise ValueError("encoder_lr_scale must lie in (0,1]")
        self.encoder_lr_scale = encoder_lr_scale
        self.action_dim = action_dim
        self.group_builder = GroupBuilderConfig(**group_builder).get_group_builder()
        self.flatten_observation = feature_extractor is None
        self.features = (
            nn.Sequential(nn.Linear(observation_dim, hidden_dim), nn.GELU())
            if feature_extractor is None
            else ModelConfig(**feature_extractor).get_model()
        )
        self.hidden_dim = hidden_dim
        self.history = (
            nn.GRUCell(hidden_dim, hidden_dim, bias=history_bias)
            if history_encoder is None else ModelConfig(**history_encoder).get_model()
        )
        self.codec = ScalarMessageCodec(
            hidden_dim, message_dim, bits,
            encoder=None if codec_encoder is None else ModelConfig(**codec_encoder).get_model(),
            decoder=None if codec_decoder is None else ModelConfig(**codec_decoder).get_model(),
        )
        if min(hidden_dim, heads, slots, inducing_points) < 1:
            raise ValueError("Group attention dimensions must be positive")
        self.fusion = GroupSetFusion(hidden_dim, heads, slots, inducing_points)
        self.reader = nn.MultiheadAttention(
            hidden_dim, heads if reader_heads is None else reader_heads,
            batch_first=True, dropout=0
        )
        self.local_head = (
            nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU())
            if local_head is None else ModelConfig(**local_head).get_model()
        )
        self.action_head = (
            nn.Linear(hidden_dim, action_dim)
            if action_head is None else ModelConfig(**action_head).get_model()
        )
        for network in (self.features, self.history, self.codec, self.local_head, self.action_head):
            for module in network.modules():
                if isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.RReLU)):
                    raise ValueError("Communication models require deterministic per-sample layers")
                if (
                    isinstance(module, (nn.MultiheadAttention, nn.RNNBase))
                    and module.dropout > 0
                ) or (
                    isinstance(module, nn.modules.dropout._DropoutNd) and module.p > 0
                ):
                    raise ValueError("Communication dropout must be zero for PPO replay")
        self.reset()

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
        if x.shape != (b * n * t, self.hidden_dim):
            raise ValueError("Feature extractor must output one hidden_dim vector per frame")
        x = x.reshape(b * n, t, self.hidden_dim)
        padding = traj_padding_mask.reshape(b * n, t)
        if isinstance(self.history, nn.GRUCell):
            h = x.new_zeros(b * n, self.hidden_dim)
            for step in range(t):
                candidate = self.history(x[:, step], h)
                h = torch.where(padding[:, step, None], h, candidate)
        else:
            # Keep one valid key for empty histories so attention softmax stays finite.
            empty = padding.all(dim=1)
            safe_padding = padding.clone()
            safe_padding[:, 0] = safe_padding[:, 0] & ~empty
            h = self.history(x.masked_fill(padding.unsqueeze(-1), 0), padding_mask=safe_padding)
            if h.shape != (b * n, self.hidden_dim):
                raise ValueError("History encoder must output one hidden_dim vector per sender")
            h = h.masked_fill(empty.unsqueeze(-1), 0)
        h = h.reshape(b, n, self.hidden_dim)
        if self.message_mode == "quantized":
            decoded, codes = self.codec(h)
            if decoded.shape != h.shape:
                raise ValueError("Codec decoder must output one hidden_dim vector per sender")
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
        tokens = decoded
        if message_override is not None:
            if (
                message_override.shape != h.shape
                or replace_mask is None
                or replace_mask.shape != alive_mask.shape
            ):
                raise ValueError("Replacement messages and mask must match sender axes")
            tokens = torch.where(replace_mask.unsqueeze(-1), message_override, tokens)

        memory, valid = self.fusion(tokens, group_indices, alive_mask)
        indices = torch.where(alive_mask, group_indices, 0)
        rows = torch.arange(b, device=h.device)[:, None]
        agent_memory = memory[rows, indices].flatten(0, 1)
        read, _ = self.reader(
            h.reshape(b * n, 1, -1), agent_memory, agent_memory, need_weights=False
        )
        features = self.local_head(torch.cat((h, read.reshape(b, n, -1)), dim=-1))
        values = self.action_head(features)
        if values.shape != (b, n, self.action_dim):
            raise ValueError("Action head must output action_dim logits per agent")
        values = values.masked_fill(~alive_mask.unsqueeze(-1), 0)
        return {
            "action_logits": values,
            "message_codes": codes,
            "group_indices": group_indices,
            "raw_tokens": h,
            "decoded_tokens": decoded,
            "group_tokens": memory,
            "group_valid": valid,
            "relation_context": memory.mean(-2),
        }

    def training_forward(self, observations, padding, alive, batch, next_state=False):
        key = "next_group_indices" if next_state else "group_indices"
        if key not in batch:
            raise ValueError(
                f"Missing {key}; select the group communication rollout profile"
            )
        groups = batch[key][:, -1].to(device=observations.device, dtype=torch.long)
        return self(observations, padding, alive, groups)

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
        groups = self.group_builder(
            torch.as_tensor(state, device=self.device).unsqueeze(0)
        ).to(device=self.device, dtype=torch.long)
        groups = groups.masked_fill(~alive, -1)
        ret = self(obs, padding, alive, groups)
        values = ret["action_logits"][0]
        masks = None
        if isinstance(next(iter(avail_actions.values())), np.ndarray):
            masks = torch.as_tensor(
                np.stack([avail_actions[a] for a in names]),
                device=self.device,
                dtype=torch.bool,
            )
        dist = masked_categorical(values, masks, active_mask=alive[0])
        actions = dist.logits.argmax(-1) if greedy_evaluation else dist.sample()
        actions = actions.masked_fill(~alive[0], 0)
        all_actions = dict(zip(names, actions.cpu().tolist()))
        result = {
            "actions": {a: all_actions[a] for a in alive_agents},
            "all_actions": all_actions,
            "all_group_indices": dict(zip(names, groups[0].cpu().tolist())),
            "message_codes": dict(zip(names, ret["message_codes"][0].cpu().numpy())),
            "snapshot_steps": {a: self.snapshot_step for a in names},
        }
        result["all_log_probs"] = dict(
            zip(names, dist.log_prob(actions).masked_fill(~alive[0], 0).cpu().tolist())
        )
        # Serialize the actual payload for byte accounting; transport remains synchronous.
        from marlite.util.communication_protocol import MessagePacket, snapshot_packets

        current_groups = groups[0].cpu().tolist()
        if self._last_groups is not None and current_groups != self._last_groups:
            self.group_version += 1
        self._last_groups = current_groups
        packets = []
        sizes = dict.fromkeys(names, 0)
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
