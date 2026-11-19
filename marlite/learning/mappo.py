"""All-timestep PPO with fixed rollout labels and explicit recurrent replay."""

from dataclasses import replace
import math

import torch

from marlite.algorithm.sequence_agent_group import DecisionInput
from marlite.data import collate_sequences, SequenceSlice
from marlite.util.action_distribution import masked_categorical
from marlite.util.return_estimation import generalized_advantage_estimation
from .backend import ExecutionBackend
from .sequence import initial_states, validate_optimizer_ownership


def prepare_ppo_targets(episodes, gamma=.99, gae_lambda=.95, normalize=True,
                        reward_aggregation="sum"):
    """Attach new detached label tensors; never mutate recorded behavior data."""
    if reward_aggregation not in ("sum", "mean"):
        raise ValueError("Reward aggregation must be sum or mean")
    result = {}
    for index, episode in episodes.items():
        episode.validate()
        if "old_value" not in episode.frames or "old_log_prob" not in episode.transitions:
            raise ValueError("PPO requires recorded old_value and old_log_prob")
        reward = episode.transitions["rewards"]
        reward = reward.sum(-1) if reward_aggregation == "sum" else reward.mean(-1)
        bootstrap, trace = episode.return_masks()
        values = episode.frames["old_value"]
        advantages, targets = generalized_advantage_estimation(
            reward, values[:-1], values[1:], bootstrap, trace, gamma, gae_lambda)
        transitions = dict(episode.transitions, advantages=advantages, value_targets=targets)
        result[index] = replace(episode, transitions=transitions)
    if normalize and result:
        labels = torch.cat([e.transitions["advantages"][e.frames["alive"][:-1].any(-1)]
                            for e in result.values()])
        if labels.numel():
            mean, std = labels.mean(), labels.std(unbiased=False).clamp_min(1e-8)
            for episode in result.values():
                episode.transitions["advantages"] = (episode.transitions["advantages"] - mean) / std
    return result


class MAPPOLearner:
    """Own PPO models/optimizers; Trainer owns sampling and PPO epochs.

    cached_burnin uses behavior state only at the warmup boundary. Exact mode
    recomputes the entire prefix for EACH update under the current parameters.
    Warmup has no autograd graph; each learning timestep contributes its loss.
    """

    def __init__(self, actor, critic, actor_optimizer, critic_optimizer, *,
                 backend=None, clip_epsilon=.2, entropy_coef=.01, vf_coef=.5,
                 max_grad_norm=5., burn_in=8, state_strategy="cached_burnin"):
        for name, value in (("clip_epsilon", clip_epsilon), ("entropy_coef", entropy_coef),
                            ("vf_coef", vf_coef), ("max_grad_norm", max_grad_norm)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if max_grad_norm == 0 or type(burn_in) is not int or burn_in < 0:
            raise ValueError("Invalid gradient clipping or warmup length")
        if state_strategy not in ("cached_burnin", "zero_burnin", "prefix_recompute"):
            raise ValueError("Unknown recurrent replay strategy")
        if actor.output_kind != "logits":
            raise ValueError("MAPPO requires a categorical logits actor")
        if {id(p) for p in actor.parameters()} & {id(p) for p in critic.parameters()}:
            raise ValueError("Actor and critic optimizers must own disjoint parameters")
        validate_optimizer_ownership(((actor, actor_optimizer), (critic, critic_optimizer)))
        self.actor, self.critic = actor, critic
        self.actor_optimizer, self.critic_optimizer = actor_optimizer, critic_optimizer
        self.backend = backend or ExecutionBackend(next(actor.parameters()).device)
        self.clip_epsilon, self.entropy_coef, self.vf_coef = clip_epsilon, entropy_coef, vf_coef
        self.max_grad_norm, self.burn_in = max_grad_norm, burn_in
        self.state_strategy = state_strategy
        self.actor_updates = self.critic_updates = self.updates = 0
        self.actor.to(self.backend.device)
        self.critic.to(self.backend.device)
        self.backend.synchronize(self.actor)
        self.backend.synchronize(self.critic)

    def _initial_states(self, episodes, sources, *, critic=False):
        return initial_states(
            self.critic if critic else self.actor, episodes, sources,
            device=self.backend.device, strategy=self.state_strategy, burn_in=self.burn_in,
            team=critic, cache_prefix="critic_state." if critic else "actor_state.",
        )

    def losses(self, episodes, sources):
        empty = not sources
        if empty:
            if not self.backend.distributed:
                raise ValueError("A PPO batch requires at least one sequence")
            # Empty ranks execute the same graph/collectives without contributing samples.
            sources = [SequenceSlice(next(iter(episodes)), 0, 1)]
        versions = {episodes[s.episode_id].policy_version for s in sources}
        if len(versions) != 1:
            raise ValueError("PPO cannot mix behavior policy versions")
        batch = collate_sequences(episodes, [SequenceSlice(s.episode_id, s.start, s.length) for s in sources])
        batch = batch.to(self.backend.device)
        if empty:
            batch.loss_mask.zero_()
        if not {"advantages", "value_targets"}.issubset(batch.transitions):
            raise ValueError("Freeze full-episode PPO targets before updating")
        actor_state = self._initial_states(episodes, sources)
        critic_state = self._initial_states(episodes, sources, critic=True)
        alive = batch.frames["alive"][:, :-1]
        output = self.actor.forward_sequence(DecisionInput(
            batch.frames["observations"][:, :-1], alive,
            batch.frames["agent_reset"][:, :-1]), actor_state)
        team_alive = alive.any(-1) & batch.filled
        team_reset = torch.zeros_like(batch.filled)
        team_reset[:, 0] = torch.tensor([s.start == 0 for s in sources], device=self.backend.device)
        values = self.critic.forward_sequence(batch.frames["states"][:, :-1], critic_state,
                                               team_alive, team_reset).values
        dist = masked_categorical(output.values, batch.frames["available_actions"][:, :-1],
                                  active_mask=alive, actions=batch.transitions["actions"])
        log_prob = dist.log_prob(batch.transitions["actions"])
        old = batch.transitions["old_log_prob"]
        ratio = (log_prob - old).exp()
        advantage = batch.transitions["advantages"][..., None]
        first = ratio * advantage
        second = ratio.clamp(1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage
        actor_mask = batch.actor_mask
        value_mask = batch.loss_mask & team_alive
        actor_count = self.backend.count(actor_mask)
        value_count = self.backend.count(value_mask)
        actor_sum = -torch.minimum(first, second)[actor_mask].sum()
        entropy_sum = dist.entropy()[actor_mask].sum()
        value_sum = (values - batch.transitions["value_targets"]).square()[value_mask].sum()
        policy_loss = (actor_sum - self.entropy_coef * entropy_sum) / actor_count.clamp_min(1)
        critic_loss = self.vf_coef * value_sum / value_count.clamp_min(1)
        with torch.no_grad():
            difference = log_prob - old
            stats = {
                "actor_loss": self.backend.sum(actor_sum) / actor_count.clamp_min(1),
                "critic_loss": self.backend.sum(value_sum) / value_count.clamp_min(1),
                "entropy": self.backend.sum(entropy_sum) / actor_count.clamp_min(1),
                "approx_kl": self.backend.sum(((ratio - 1) - difference)[actor_mask].sum()) / actor_count.clamp_min(1),
                "clip_fraction": self.backend.sum(((ratio - 1).abs() > self.clip_epsilon)[actor_mask].float().sum()) / actor_count.clamp_min(1),
                "agent_steps": actor_count, "team_steps": value_count,
            }
        return policy_loss, critic_loss, {k: float(v) for k, v in stats.items()}

    def update(self, episodes, sources):
        self.actor.train()
        self.critic.train()
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        policy_loss, critic_loss, metrics = self.losses(episodes, sources)
        if not torch.isfinite(policy_loss) or not torch.isfinite(critic_loss):
            raise RuntimeError("Nonfinite PPO loss")
        policy_loss.backward()
        if self.vf_coef:
            critic_loss.backward()
        self.backend.reduce_gradients(self.actor.parameters())
        self.backend.reduce_gradients(self.critic.parameters())
        if metrics["agent_steps"]:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            self.actor_updates += 1
        if self.vf_coef and metrics["team_steps"]:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            self.critic_updates += 1
        if metrics["agent_steps"] or (self.vf_coef and metrics["team_steps"]):
            self.updates += 1
        return metrics

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor_updates": self.actor_updates, "critic_updates": self.critic_updates,
            "updates": self.updates,
        }

    def load_state_dict(self, state):
        for name in ("actor", "critic", "actor_optimizer", "critic_optimizer"):
            getattr(self, name).load_state_dict(state[name])
        for name in ("actor_updates", "critic_updates", "updates"):
            setattr(self, name, state[name])
