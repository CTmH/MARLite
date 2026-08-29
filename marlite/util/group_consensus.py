"""Shared validation helpers for fixed-capacity group consensus tensors."""


def validate_group_capacity(agent_group, data_constructor):
    """Ensure group-aware agents and SSL constructors share a group axis.

    Legacy non-group constructors do not expose ``n_groups``; they are left
    unchanged and their ordinary tensor-shape validation remains authoritative.
    """
    agent_capacity = getattr(agent_group, "n_groups", None)
    target_capacity = getattr(data_constructor, "n_groups", None)

    if target_capacity is None:
        return agent_capacity
    if agent_capacity is None:
        raise ValueError(
            "A group-aware SSL data constructor requires its agent group "
            "builder to expose a fixed n_groups capacity"
        )
    if int(agent_capacity) != int(target_capacity):
        raise ValueError(
            "Group capacity mismatch: "
            f"agent_group n_groups={agent_capacity}, "
            f"data_constructor n_groups={target_capacity}"
        )
    return int(agent_capacity)


def validate_group_reconstruction_shapes(consensus, targets, construct_mask) -> int:
    """Validate the shared group dimension before reconstruction decoding."""
    consensus_groups = consensus.shape[1]
    target_groups = targets.shape[1]
    mask_groups = construct_mask.shape[1]
    if not (consensus_groups == target_groups == mask_groups):
        raise ValueError(
            "Inconsistent group capacity for reconstruction: "
            f"consensus={consensus_groups}, targets={target_groups}, "
            f"construct_mask={mask_groups}"
        )
    return consensus_groups
