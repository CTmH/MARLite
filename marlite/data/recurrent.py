"""Explicit truncated-BPTT state reconstruction for contiguous chunks."""

import torch

from marlite.algorithm.temporal import detach_state


def reconstruct_state(encoder, inputs, valid, reset, learning_start, *,
                      strategy="cached_burnin", burn_in=8, cached_state=None):
    """Return a detached state immediately before learning_start.

    Inputs cover the episode prefix from reset. cached_state is the behavior
    state BEFORE max(0, learning_start-burn_in), not the state at the first
    learning frame. Its version/identity must be checked by the calling learner.
    prefix_recompute uses current parameters and ignores behavior-state caches.
    """
    if (type(learning_start) is not int or not 0 <= learning_start <= inputs.shape[1]
            or type(burn_in) is not int or burn_in < 0):
        raise ValueError("Invalid state reconstruction interval")
    if strategy not in ("cached_burnin", "zero_burnin", "prefix_recompute"):
        raise ValueError("Unknown state reconstruction strategy")
    begin = 0 if strategy == "prefix_recompute" else max(0, learning_start - burn_in)
    if strategy == "cached_burnin":
        if cached_state is None:
            raise ValueError("cached_burnin requires the state at the warmup boundary")
        state = detach_state(cached_state)
    else:
        state = encoder.initial_state(inputs.shape[0], inputs.shape[2], inputs.device, inputs.dtype)
    with torch.no_grad():
        if begin < learning_start:
            state = encoder.forward_sequence(inputs[:, begin:learning_start], state,
                                             valid[:, begin:learning_start],
                                             reset[:, begin:learning_start]).state
    return detach_state(state)
