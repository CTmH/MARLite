"""Training-only, state-independent spatial query sampling for complementary pursuit."""

import math
import torch
from marlite.environment.complementary_events import ATTACK_OFFSETS


class ComplementaryAttackQueries:
    """Sample existence, complementary-hit and capture queries (14 input features).

    Candidate attackers occupy specified cells and attack the specified target.
    Unlisted agents stay. Coordinates are drawn independently of target presence;
    labels describe pre-movement attacks, not a policy's future capture probability.
    State schema is H,W,C with evader presence/HP at 1/2 and pursuer presence at 3.
    """

    query_dim = 14

    def __init__(
        self, n_queries=64, evader_hp=5.0, effective_damage=10.0, ineffective_damage=0.2
    ):
        if type(n_queries) is not int or n_queries < 1:
            raise ValueError("n_queries must be positive")
        if (
            not all(
                math.isfinite(v)
                for v in (evader_hp, effective_damage, ineffective_damage)
            )
            or evader_hp <= 0
            or effective_damage < 0
            or ineffective_damage < 0
        ):
            raise ValueError("Invalid HP or damage")
        self.n_queries = n_queries
        self.evader_hp = evader_hp
        self.effective_damage = effective_damage
        self.ineffective_damage = ineffective_damage

    def __call__(self, batch, valid, device):
        state = batch["states"][:, -1].to(device)
        if state.ndim != 4 or state.shape[-1] != 25:
            raise ValueError("Complementary labels require the native 25-channel state")
        b, g = valid.shape
        width, height = state.shape[1:3]
        shape = (b, g, self.n_queries)
        target = torch.stack(
            [
                torch.randint(width, shape, device=device),
                torch.randint(height, shape, device=device),
            ],
            -1,
        )
        offsets = torch.tensor(ATTACK_OFFSETS, device=device)
        positions = (
            target.unsqueeze(-2)
            + offsets[torch.randint(12, (*shape, 3), device=device)]
        )
        modes = torch.randint(2, (*shape, 3), device=device)
        counts = torch.randint(1, 4, shape, device=device)
        active = torch.arange(3, device=device) < counts.unsqueeze(-1)
        kind = torch.randint(3, shape, device=device)
        return self.evaluate(state, valid, target, positions, modes, active, kind)

    def evaluate(self, state, valid, target, positions, modes, active, kind):
        """Deterministic label path, also used to compare hand-built native scenes."""
        width, height = state.shape[1:3]
        rows = torch.arange(state.shape[0], device=state.device)[:, None, None]
        exists = state[rows, target[..., 0], target[..., 1], 1] > 0
        hp = state[rows, target[..., 0], target[..., 1], 2] * self.evader_hp
        inside = (
            (positions[..., 0] >= 0)
            & (positions[..., 0] < width)
            & (positions[..., 1] >= 0)
            & (positions[..., 1] < height)
        )
        x = positions[..., 0].clamp(0, width - 1)
        y = positions[..., 1].clamp(0, height - 1)
        present = (state[rows.unsqueeze(-1), x, y, 3] > 0) & inside & active
        effective = (present & (modes == 0)).any(-1) & (present & (modes == 1)).any(-1)
        captured = present.any(-1) & (
            hp <= torch.where(effective, self.effective_damage, self.ineffective_damage)
        )
        labels = torch.where(
            kind == 0, exists, exists & torch.where(kind == 1, effective, captured)
        ).float()
        unique = torch.ones_like(exists)
        for i in range(3):
            for j in range(i):
                unique &= ~(
                    (positions[..., i, :] == positions[..., j, :]).all(-1)
                    & active[..., i]
                    & active[..., j]
                )
        mask = valid.unsqueeze(-1) & ((kind == 0) | unique)
        damage = torch.where(effective, self.effective_damage, self.ineffective_damage)
        tolerance = (
            8
            * torch.finfo(torch.float32).eps
            * torch.maximum(hp.abs(), damage).clamp_min(1)
        )
        ambiguous = exists & present.any(-1) & ((hp - damage).abs() <= tolerance)
        mask &= ~((kind == 2) & ambiguous)
        scale = state.new_tensor([max(width - 1, 1), max(height - 1, 1)])
        actor_features = torch.cat(
            [positions / scale, torch.where(active, modes, -1).unsqueeze(-1)], -1
        )
        queries = torch.cat(
            [
                torch.nn.functional.one_hot(kind, 3),
                target / scale,
                actor_features.flatten(-2),
            ],
            -1,
        ).float()
        return {
            "relation_queries": queries,
            "relation_labels": labels,
            "relation_mask": mask,
        }
