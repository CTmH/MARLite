"""Offline sender-replacement warm-up, separate from on-policy rollout data."""

from copy import deepcopy
import torch
from torch import nn
from marlite.util.action_distribution import masked_categorical


def action_distillation(student, teacher, alive, output_key, available=None):
    if output_key == "action_logits":
        p = masked_categorical(teacher, available, active_mask=alive)
        q = masked_categorical(student, available, active_mask=alive)
        loss = torch.distributions.kl_divergence(p, q)
    else:
        mask = alive.unsqueeze(-1).expand_as(student)
        if available is not None:
            mask = mask & available
        delta = torch.where(mask, student - teacher, 0.0)
        return delta.square().sum() / mask.sum().clamp_min(1)
    return (loss * alive).sum() / alive.sum().clamp_min(1)


class CommunicationCurriculum(nn.Module):
    """Owns a frozen raw teacher and resumable single/half/all replacement stages.

    The caller owns the optimizer and saves it alongside this state_dict.
    Pass an already trained teacher; constructing this class does not train it.
    """

    def __init__(self, student, teacher, stage_steps=(10000, 10000, 10000)):
        super().__init__()
        if (
            student.message_mode != "quantized"
            or student.output_key != teacher.output_key
            or teacher.message_mode != "raw"
        ):
            raise ValueError("Student and raw teacher must use the same RL algorithm")
        if len(stage_steps) != 3 or any(s < 1 for s in stage_steps):
            raise ValueError("Provide three positive warm-up stage lengths")
        if student.agent_model_dict != teacher.agent_model_dict:
            raise ValueError("Teacher and student sender schemas must match")
        if student.mask_observation_channels != teacher.mask_observation_channels:
            raise ValueError(
                "Teacher and student must use the same legal observation channels"
            )
        self.student = student
        self.teacher = deepcopy(teacher).requires_grad_(False).eval()
        self.register_buffer("stage_steps", torch.tensor(stage_steps, dtype=torch.long))
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("readout_updates", torch.zeros((), dtype=torch.long))

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    @property
    def stage(self):
        return int((self.updates >= self.stage_steps.cumsum(0)).sum().item())

    def replacement_mask(self, groups, alive):
        stage = self.stage
        if stage >= 3:
            raise RuntimeError(
                "Warm-up is complete; start fresh fully quantized rollouts"
            )
        selected = torch.zeros_like(alive)
        for b in range(groups.shape[0]):
            for g in groups[b, alive[b]].unique():
                indices = torch.where(alive[b] & (groups[b] == g))[0]
                count = (
                    1
                    if stage == 0
                    else (len(indices) + 1) // 2
                    if stage == 1
                    else len(indices)
                )
                if stage == 0:
                    index = int(self.updates.item()) % len(indices)
                    indices = indices[index : index + 1]
                else:
                    indices = indices[
                        torch.randperm(len(indices), device=indices.device)[:count]
                    ]
                selected[b, indices] = True
        return selected

    def loss(self, observations, padding, alive, groups, available=None, batch=None):
        selected = self.replacement_mask(groups, alive)
        with torch.no_grad():
            target = self.teacher(observations, padding, alive, groups)
        student = self.student(observations, padding, alive, groups)
        # Frozen parameters still propagate gradients to replacement tokens.
        mixed = self.teacher(
            observations,
            padding,
            alive,
            groups,
            message_override=student["decoded_tokens"],
            replace_mask=selected,
        )
        loss = action_distillation(
            mixed[self.student.output_key],
            target[self.student.output_key],
            alive,
            self.student.output_key,
            available,
        )
        if batch is not None:
            relation = self.teacher.relation_loss(mixed, batch)
            if relation is not None:
                loss = loss + relation
        return loss

    def step(
        self,
        optimizer,
        observations,
        padding,
        alive,
        groups,
        available=None,
        batch=None,
    ):
        optimizer.zero_grad(set_to_none=True)
        loss = self.loss(observations, padding, alive, groups, available, batch)
        loss.backward()
        nn.utils.clip_grad_norm_(self.student.parameters(), 5.0)
        optimizer.step()
        self.updates.add_(1)
        return float(loss.detach())

    def readout_step(
        self,
        optimizer,
        observations,
        padding,
        alive,
        groups,
        available=None,
        batch=None,
    ):
        """Train the student's own fusion/NP/head on fixed learned messages.

        This explicit handoff is needed when the student differs from the teacher.
        It does not update message encoders, even if the optimizer includes them.
        """
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            target = self.teacher(observations, padding, alive, groups)
        student = self.student(
            observations, padding, alive, groups, detach_messages=True
        )
        loss = action_distillation(
            student[self.student.output_key],
            target[self.student.output_key],
            alive,
            self.student.output_key,
            available,
        )
        if batch is not None:
            relation = self.student.relation_loss(student, batch)
            if relation is not None:
                loss = loss + relation
        loss.backward()
        nn.utils.clip_grad_norm_(self.student.parameters(), 5.0)
        optimizer.step()
        self.readout_updates.add_(1)
        return float(loss.detach())

    def save_checkpoint(self, path, optimizer):
        """Save warm-up/readout parameters, optimizer and sampling RNG together."""
        state = {
            "model": self.state_dict(),
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda_rng"] = torch.cuda.get_rng_state_all()
        if torch.backends.mps.is_available():
            state["mps_rng"] = torch.mps.get_rng_state()
        torch.save(state, path)

    def load_checkpoint(self, path, optimizer):
        """Restore into an instance built from the same saved experiment config."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        if "cuda_rng" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        if "mps_rng" in state and torch.backends.mps.is_available():
            torch.mps.set_rng_state(state["mps_rng"])
