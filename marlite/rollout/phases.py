"""Pluggable rollout collection phases.

The for-loop inside ``persistent_env_rollout`` and ``multiprocess_rollout``
is decomposed into five *collection phases*.  Each phase is a pure function
that receives the episode dict and a context dictionary, and appends its
attributes to the episode.

Pre-built ``RolloutPhases`` instances are provided for every algorithm
profile and can be passed directly to the rollout workers (they are
pickle-safe because they only reference module-level functions).

When ``required_attrs`` is a **list** of explicit attribute names (instead
of a predefined profile name), ``resolve_phases`` builds filtered phase
functions via :func:`functools.partial` — each wrapped function only
collects the attributes present in the list.
"""

from functools import partial
from typing import Any, Callable, Dict, FrozenSet, List, NamedTuple, Optional, Union

# ---------------------------------------------------------------------------
# Phase type
# ---------------------------------------------------------------------------

CollectFn = Callable[[Dict, Dict], None]
"""A collection-phase function: ``(episode_dict, ctx_dict) -> None``."""


class RolloutPhases(NamedTuple):
    """Bundle of collection-phase functions for one algorithm profile."""
    pre_step:  CollectFn   # Before env.step() — observation-phase attrs
    post_step: CollectFn   # After  env.step() — rewards, terms, next_obs, ...
    terminal:  CollectFn   # When the episode terminates — next-* defaults
    next_attr: CollectFn   # After  agent.act() — next_alive_mask, next_avail_actions, ...
    finalize:  CollectFn   # After the episode loop — win_tag, length, reward


# ===================================================================
# Atomic phase functions (module-level → pickle-safe)
# ===================================================================

# -------------------------------------------------------------------
# pre_step — collects observation-phase attrs
# -------------------------------------------------------------------

def _pre_essential(ep: Dict, ctx: Dict) -> None:
    ep["alive_mask"].append(ctx["alive_mask"])
    ep["observations"].append(ctx["observations"])
    ep["states"].append(ctx["states"])
    ep["actions"].append(ctx["actions"])
    ep["avail_actions"].append(ctx["avail_actions"])
    ep["infos"].append(ctx["infos"])


def _pre_mappo(ep: Dict, ctx: Dict) -> None:
    ep["all_log_probs"].append(ctx["all_log_probs"])
    ep["log_probs"].append(ctx["log_probs"])


def _pre_graph(ep: Dict, ctx: Dict) -> None:
    ep["edge_indices"].append(ctx["edge_indices"])


def _pre_group(ep: Dict, ctx: Dict) -> None:
    ep["group_indices"].append(ctx["group_indices"])


# -------------------------------------------------------------------
# post_step — collects env.step() result attrs
# -------------------------------------------------------------------

def _post_essential(ep: Dict, ctx: Dict) -> None:
    ep["rewards"].append(ctx["rewards"])
    ep["terminations"].append(ctx["terminations"])
    ep["truncations"].append(ctx["truncations"])
    ep["next_observations"].append(ctx["observations"])
    ep["all_agents_sum_rewards"].append(ctx["all_agents_sum_rewards"])


# -------------------------------------------------------------------
# terminal — collects next-* attrs (defaults / reuse) on early exit
# -------------------------------------------------------------------

def _term_essential(ep: Dict, ctx: Dict) -> None:
    ep["next_avail_actions"].append(ctx["default_avail_actions"])
    ep["next_alive_mask"].append(ctx["default_alive_mask"])


def _term_graph(ep: Dict, ctx: Dict) -> None:
    ep["next_edge_indices"].append(ctx["edge_indices"])


def _term_group(ep: Dict, ctx: Dict) -> None:
    ep["next_group_indices"].append(ctx["group_indices"])


# -------------------------------------------------------------------
# next_attr — collects next-* attrs (actual values) after agent.act()
# -------------------------------------------------------------------

def _next_essential(ep: Dict, ctx: Dict) -> None:
    ep["next_alive_mask"].append(ctx["alive_mask"])
    ep["next_avail_actions"].append(ctx["avail_actions"])


def _next_graph(ep: Dict, ctx: Dict) -> None:
    ep["next_edge_indices"].append(ctx["edge_indices"])


def _next_group(ep: Dict, ctx: Dict) -> None:
    ep["next_group_indices"].append(ctx["group_indices"])


# -------------------------------------------------------------------
# finalize — sets episode-level attrs after loop exit
# -------------------------------------------------------------------

def _finalize_essential(ep: Dict, ctx: Dict) -> None:
    ep["win_tag"] = ctx["win_tag"]
    ep["episode_length"] = ctx["episode_length"]
    ep["episode_reward"] = ctx["episode_reward"]


# ===================================================================
# Composed phase functions (no lambdas — pickle-safe)
# ===================================================================

# --- pre_step compositions ---

def pre_step_full(ep: Dict, ctx: Dict) -> None:
    _pre_essential(ep, ctx)
    _pre_mappo(ep, ctx)
    _pre_graph(ep, ctx)
    _pre_group(ep, ctx)


def pre_step_qmix(ep: Dict, ctx: Dict) -> None:
    _pre_essential(ep, ctx)


def pre_step_mappo(ep: Dict, ctx: Dict) -> None:
    _pre_essential(ep, ctx)
    _pre_mappo(ep, ctx)


def pre_step_graph_qmix(ep: Dict, ctx: Dict) -> None:
    _pre_essential(ep, ctx)
    _pre_graph(ep, ctx)


def pre_step_graph_mappo(ep: Dict, ctx: Dict) -> None:
    _pre_essential(ep, ctx)
    _pre_mappo(ep, ctx)
    _pre_graph(ep, ctx)


def pre_step_group_consensus(ep: Dict, ctx: Dict) -> None:
    _pre_essential(ep, ctx)
    _pre_group(ep, ctx)


# --- terminal compositions ---

def terminal_full(ep: Dict, ctx: Dict) -> None:
    _term_essential(ep, ctx)
    _term_graph(ep, ctx)
    _term_group(ep, ctx)


def terminal_essential(ep: Dict, ctx: Dict) -> None:
    _term_essential(ep, ctx)


def terminal_graph(ep: Dict, ctx: Dict) -> None:
    _term_essential(ep, ctx)
    _term_graph(ep, ctx)


def terminal_group(ep: Dict, ctx: Dict) -> None:
    _term_essential(ep, ctx)
    _term_group(ep, ctx)


# --- next_attr compositions ---

def next_attr_full(ep: Dict, ctx: Dict) -> None:
    _next_essential(ep, ctx)
    _next_graph(ep, ctx)
    _next_group(ep, ctx)


def next_attr_essential(ep: Dict, ctx: Dict) -> None:
    _next_essential(ep, ctx)


def next_attr_graph(ep: Dict, ctx: Dict) -> None:
    _next_essential(ep, ctx)
    _next_graph(ep, ctx)


def next_attr_group(ep: Dict, ctx: Dict) -> None:
    _next_essential(ep, ctx)
    _next_group(ep, ctx)


# ===================================================================
# Generic filtered phase functions (module-level → pickle-safe).
#
# Each function takes an *attrs_set* that controls which attributes are
# actually collected.  :func:`_build_custom_phases` wraps them with
# :func:`functools.partial` so the resulting ``RolloutPhases`` is
# pickle-safe.
# ===================================================================

def _pre_step_filtered(ep: Dict, ctx: Dict, attrs_set: FrozenSet[str]) -> None:
    """pre_step that only collects attributes listed in *attrs_set*."""
    if "alive_mask" in attrs_set:
        ep["alive_mask"].append(ctx["alive_mask"])
    if "observations" in attrs_set:
        ep["observations"].append(ctx["observations"])
    if "states" in attrs_set:
        ep["states"].append(ctx["states"])
    if "actions" in attrs_set:
        ep["actions"].append(ctx["actions"])
    if "avail_actions" in attrs_set:
        ep["avail_actions"].append(ctx["avail_actions"])
    if "infos" in attrs_set:
        ep["infos"].append(ctx["infos"])
    if "edge_indices" in attrs_set:
        ep["edge_indices"].append(ctx["edge_indices"])
    if "group_indices" in attrs_set:
        ep["group_indices"].append(ctx["group_indices"])
    if "all_log_probs" in attrs_set:
        ep["all_log_probs"].append(ctx["all_log_probs"])
    if "log_probs" in attrs_set:
        ep["log_probs"].append(ctx["log_probs"])


def _post_step_filtered(ep: Dict, ctx: Dict, attrs_set: FrozenSet[str]) -> None:
    """post_step that only collects attributes listed in *attrs_set*."""
    if "rewards" in attrs_set:
        ep["rewards"].append(ctx["rewards"])
    if "terminations" in attrs_set:
        ep["terminations"].append(ctx["terminations"])
    if "truncations" in attrs_set:
        ep["truncations"].append(ctx["truncations"])
    if "next_observations" in attrs_set:
        ep["next_observations"].append(ctx["observations"])
    if "all_agents_sum_rewards" in attrs_set:
        ep["all_agents_sum_rewards"].append(ctx["all_agents_sum_rewards"])


def _terminal_filtered(ep: Dict, ctx: Dict, attrs_set: FrozenSet[str]) -> None:
    """terminal that only collects attributes listed in *attrs_set*."""
    if "next_avail_actions" in attrs_set:
        ep["next_avail_actions"].append(ctx["default_avail_actions"])
    if "next_alive_mask" in attrs_set:
        ep["next_alive_mask"].append(ctx["default_alive_mask"])
    if "next_edge_indices" in attrs_set:
        ep["next_edge_indices"].append(ctx["edge_indices"])
    if "next_group_indices" in attrs_set:
        ep["next_group_indices"].append(ctx["group_indices"])


def _next_attr_filtered(ep: Dict, ctx: Dict, attrs_set: FrozenSet[str]) -> None:
    """next_attr that only collects attributes listed in *attrs_set*."""
    if "next_alive_mask" in attrs_set:
        ep["next_alive_mask"].append(ctx["alive_mask"])
    if "next_avail_actions" in attrs_set:
        ep["next_avail_actions"].append(ctx["avail_actions"])
    if "next_edge_indices" in attrs_set:
        ep["next_edge_indices"].append(ctx["edge_indices"])
    if "next_group_indices" in attrs_set:
        ep["next_group_indices"].append(ctx["group_indices"])


def _finalize_filtered(ep: Dict, ctx: Dict, attrs_set: FrozenSet[str]) -> None:
    """finalize that only sets attributes listed in *attrs_set*."""
    if "win_tag" in attrs_set:
        ep["win_tag"] = ctx["win_tag"]
    if "episode_length" in attrs_set:
        ep["episode_length"] = ctx["episode_length"]
    if "episode_reward" in attrs_set:
        ep["episode_reward"] = ctx["episode_reward"]


def _build_custom_phases(required_attrs: List[str]) -> RolloutPhases:
    """Build a ``RolloutPhases`` that only collects *required_attrs*.

    Uses :func:`functools.partial` to bind a frozen set of attribute names
    to the filtered phase functions.  The resulting ``partial`` objects are
    pickle-safe because the underlying filtered functions are defined at
    module level.
    """
    attrs_set = frozenset(required_attrs)
    return RolloutPhases(
        pre_step=partial(_pre_step_filtered, attrs_set=attrs_set),
        post_step=partial(_post_step_filtered, attrs_set=attrs_set),
        terminal=partial(_terminal_filtered, attrs_set=attrs_set),
        next_attr=partial(_next_attr_filtered, attrs_set=attrs_set),
        finalize=partial(_finalize_filtered, attrs_set=attrs_set),
    )


# ===================================================================
# Pre-built RolloutPhases instances
# ===================================================================

FULL_PHASES = RolloutPhases(
    pre_step=pre_step_full,
    post_step=_post_essential,
    terminal=terminal_full,
    next_attr=next_attr_full,
    finalize=_finalize_essential,
)

QMIX_PHASES = RolloutPhases(
    pre_step=pre_step_qmix,
    post_step=_post_essential,
    terminal=terminal_essential,
    next_attr=next_attr_essential,
    finalize=_finalize_essential,
)

MAPPO_PHASES = RolloutPhases(
    pre_step=pre_step_mappo,
    post_step=_post_essential,
    terminal=terminal_essential,
    next_attr=next_attr_essential,
    finalize=_finalize_essential,
)

GRAPH_QMIX_PHASES = RolloutPhases(
    pre_step=pre_step_graph_qmix,
    post_step=_post_essential,
    terminal=terminal_graph,
    next_attr=next_attr_graph,
    finalize=_finalize_essential,
)

GRAPH_MAPPO_PHASES = RolloutPhases(
    pre_step=pre_step_graph_mappo,
    post_step=_post_essential,
    terminal=terminal_graph,
    next_attr=next_attr_graph,
    finalize=_finalize_essential,
)

GROUP_CONSENSUS_PHASES = RolloutPhases(
    pre_step=pre_step_group_consensus,
    post_step=_post_essential,
    terminal=terminal_group,
    next_attr=next_attr_group,
    finalize=_finalize_essential,
)


# ===================================================================
# Registry (str → RolloutPhases)
# ===================================================================

PHASE_REGISTRY: Dict[str, RolloutPhases] = {
    "full":              FULL_PHASES,
    "qmix":              QMIX_PHASES,
    "mappo":             MAPPO_PHASES,
    "graph_qmix":         GRAPH_QMIX_PHASES,
    "graph_mappo":       GRAPH_MAPPO_PHASES,
    "group_consensus":    GROUP_CONSENSUS_PHASES,
    "ssl_group_consensus": GROUP_CONSENSUS_PHASES,
    "vae_graph_qmix":     GRAPH_QMIX_PHASES,
    "msg_aggr_qmix":     QMIX_PHASES,
    "g2anet_mappo":      GRAPH_MAPPO_PHASES,
}


# ===================================================================
# Resolver
# ===================================================================

def resolve_phases(
    required_attrs: Optional[Union[str, List[str], tuple]] = None,
) -> RolloutPhases:
    """Return the ``RolloutPhases`` for *required_attrs*.

    ``None`` or ``"full"``
        Full profile — collects every registered attribute (backward
        compatible).

    ``str`` (profile name)
        Looks up the named profile in ``PHASE_REGISTRY`` (e.g., ``"qmix"``,
        ``"mappo"``, ``"graph_qmix"``).

    ``list[str]``
        Builds a custom ``RolloutPhases`` that collects exactly the listed
        attributes.  Every attribute name must be a known key in
        ``ATTR_REGISTRY`` (validation is performed by
        :func:`~marlite.rollout.attribute_spec.resolve_required_attrs`
        before this function is called).
    """
    if required_attrs is None or required_attrs == "full":
        return FULL_PHASES
    if isinstance(required_attrs, str):
        if required_attrs in PHASE_REGISTRY:
            return PHASE_REGISTRY[required_attrs]
        raise ValueError(
            f"Unknown phases profile '{required_attrs}'. "
            f"Available: {list(PHASE_REGISTRY.keys())}"
        )
    if isinstance(required_attrs, (list, tuple)):
        return _build_custom_phases(list(required_attrs))
    raise TypeError(
        f"required_attrs must be None, str, or list, got {type(required_attrs)}"
    )
