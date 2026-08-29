from types import SimpleNamespace

import pytest
import torch

from marlite.util.group_consensus import (
    validate_group_capacity,
    validate_group_reconstruction_shapes,
)


def test_matching_group_capacity_is_accepted():
    agent_group = SimpleNamespace(n_groups=6)
    data_constructor = SimpleNamespace(n_groups=6)
    assert validate_group_capacity(agent_group, data_constructor) == 6


def test_mismatched_group_capacity_is_rejected():
    agent_group = SimpleNamespace(n_groups=5)
    data_constructor = SimpleNamespace(n_groups=6)
    with pytest.raises(ValueError, match="agent_group n_groups=5"):
        validate_group_capacity(agent_group, data_constructor)


def test_legacy_non_group_constructor_is_unchanged():
    agent_group = SimpleNamespace(n_groups=3)
    data_constructor = SimpleNamespace()
    assert validate_group_capacity(agent_group, data_constructor) == 3


def test_group_reconstruction_shapes_accept_zero_padded_capacity():
    consensus = torch.zeros(128, 6, 64)
    targets = torch.zeros(128, 6, 5, 13, 13)
    construct_mask = torch.zeros(128, 6, dtype=torch.bool)
    construct_mask[:, :5] = True
    assert (
        validate_group_reconstruction_shapes(
            consensus, targets, construct_mask
        )
        == 6
    )


def test_group_reconstruction_shapes_report_mismatch():
    consensus = torch.zeros(128, 5, 64)
    targets = torch.zeros(128, 6, 5, 13, 13)
    construct_mask = torch.zeros(128, 6, dtype=torch.bool)
    with pytest.raises(
        ValueError,
        match="consensus=5, targets=6, construct_mask=6",
    ):
        validate_group_reconstruction_shapes(
            consensus, targets, construct_mask
        )
