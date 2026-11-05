"""Use recorded communication contexts without changing legacy agent contracts."""


def agent_training_forward(
    agent, observations, padding, alive, batch, next_state=False
):
    if hasattr(agent, "training_forward"):
        return agent.training_forward(
            observations, padding, alive, batch, next_state=next_state
        )
    return agent(observations, padding, alive)


class StateThresholdQueries:
    """Training-only binary queries of selected flat state coordinates.

    This is a generic smoke-test constructor, not a capture-event oracle.
    Environment-specific relation constructors must reproduce native rules.
    """

    def __init__(self, indices, queries, thresholds):
        import torch

        if (
            not indices
            or len(indices) != len(queries)
            or len(indices) != len(thresholds)
        ):
            raise ValueError("Each state index needs one query and threshold")
        if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in indices):
            raise ValueError("State indices must be nonnegative integers")
        self.indices = indices
        self.queries = torch.tensor(queries, dtype=torch.float32)
        self.thresholds = torch.tensor(thresholds, dtype=torch.float32)
        if (
            self.queries.ndim != 2
            or not torch.isfinite(self.queries).all()
            or not torch.isfinite(self.thresholds).all()
        ):
            raise ValueError("Queries and thresholds must be finite")

    def __call__(self, batch, valid, device):
        import torch

        state = batch["states"][:, -1].to(device).flatten(1)
        if max(self.indices) >= state.shape[-1]:
            raise ValueError("Relation state index exceeds state width")
        if not torch.isfinite(state[:, self.indices]).all():
            raise ValueError("Relation target state coordinates must be finite")
        labels = (state[:, self.indices] > self.thresholds.to(device)).float()
        b, g = valid.shape
        return {
            "relation_queries": self.queries.to(device)[None, None].expand(
                b, g, -1, -1
            ),
            "relation_labels": labels[:, None].expand(-1, g, -1),
            "relation_mask": valid.unsqueeze(-1).expand(-1, -1, len(self.indices)),
        }


def advance_communication_step(agent):
    if hasattr(agent, "advance_training_step"):
        agent.advance_training_step()


def validate_communication_result(result, required_attrs):
    from marlite.rollout.attribute_spec import resolve_required_attrs

    if required_attrs is None or required_attrs == "full":
        return
    for field in ("message_codes", "snapshot_steps"):
        if field in resolve_required_attrs(required_attrs) and field not in result:
            raise ValueError(f"Communication profile requires agent output {field}")
