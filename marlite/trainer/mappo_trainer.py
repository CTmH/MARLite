from marlite.util.communication_training import (
    agent_training_forward,
    advance_communication_step,
    message_gradient_cosine,
)
"""MAPPO Trainer — Multi-Agent PPO with on-policy training loop.

Implements the MAPPO algorithm (Yu et al., 2021) for discrete action spaces.
Uses Generalized Advantage Estimation (GAE) with a centralized value critic
and per-agent stochastic policies (categorical distributions).  Supports both
single-GPU and multi-GPU training via replicated workers.
"""

import math
import os
from pathlib import Path
from tempfile import mkdtemp
import yaml
import numpy as np
import torch
from marlite.util.return_estimation import ppo_targets
import torch.nn.functional as F
from marlite.util.action_distribution import masked_categorical
from tqdm import tqdm
from absl import logging

from marlite.trainer.onpolicy_trainer import OnPolicyTrainer
from marlite.trainer.trainer_worker_group.mappo_worker_group import MAPPOWorkerGroup
from marlite.algorithm.critic.mixer import Mixer as MixerCritic
from marlite.util.serialization import (
    serialize_to_buffer,
    deserialize_from_buffer,
    get_state_dict,
)
from marlite.util.trajectory_dataset import TrajectoryDataLoader


class MAPPOTrainer(OnPolicyTrainer):
    """MAPPO (Multi-Agent Proximal Policy Optimization) trainer.

    On-policy actor-critic algorithm with:
    - Centralized critic V(s) estimating the value of the global state.
    - Per-agent stochastic policy pi(a|o) modelled as a categorical distribution.
    - GAE for computing advantage estimates over trajectory segments.
    - PPO clipped surrogate objective for stable policy updates.

    Supports single-GPU (main process) and multi-GPU (worker processes) modes.
    In on-policy mode, the replay buffer is cleared after each iteration and
    refilled by ``evaluate()`` calls, so each batch of data is used exactly once.

    Parameters
    ----------
    clip_epsilon : float
        PPO clip range for the importance sampling ratio.
    gae_lambda : float
        Trace coefficient in [0, 1], used when advantage_estimator='gae'.
        The default one_step estimator is equivalent to lambda=0.
    entropy_coef : float
        Coefficient for the entropy bonus (encourages exploration).
    vf_coef : float
        Coefficient for the value function loss in the combined loss.
    max_grad_norm : float
        Maximum gradient norm for clipping.
    experimental_message_target_kl : float or None
        Optional post-pass policy KL threshold for single-device group MAPPO.
        Quantized actors also report hard-code changes. None preserves the standard loop.
    **kwargs :
        Forwarded to ``OnPolicyTrainer.__init__``.
    """

    def __init__(
        self,
        clip_epsilon: float = 0.2,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 5.0,
        experimental_message_target_kl: float | None = None,
        **kwargs,
    ):
        # Must be set before super().__init__() because _create_worker_group()
        # is called during _setup_multi_gpu() in OnPolicyTrainer.__init__().
        self.clip_epsilon = clip_epsilon
        # OnPolicyTrainer precomputes fixed GAE labels before PPO updates.
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.actor_auxiliary_objective = None
        if experimental_message_target_kl is not None and (
            not math.isfinite(experimental_message_target_kl)
            or experimental_message_target_kl <= 0
        ):
            raise ValueError("experimental_message_target_kl must be finite and positive")
        self.experimental_message_target_kl = experimental_message_target_kl
        super().__init__(**kwargs)
        if (
            self.experimental_message_target_kl is not None
            and getattr(self.eval_agent_group, "message_mode", None) not in ("quantized", "raw", "local")
        ):
            raise ValueError("Experimental message KL requires a group communication actor")
        if self.experimental_message_target_kl is not None:
            self._message_rollout_device = self.rolloutmanager_config.config["device"]
            if not isinstance(self._message_rollout_device, str):
                raise ValueError("Experimental message KL requires a single rollout device")
        if isinstance(self.eval_critic, MixerCritic):
            raise TypeError(
                "Critic subclass required, not Mixer subclass"
            )

        self.best_agent_group_params = serialize_to_buffer(
            get_state_dict(self.eval_agent_group)
        )
        self.best_critic_params = serialize_to_buffer(
            get_state_dict(self.eval_critic)
        )

    # ------------------------------------------------------------------
    # Best-model persistence (direct write, no eval-model restore)
    # ------------------------------------------------------------------

    def save_best_model(self):
        """Write cached best agent and critic params directly to disk."""
        best_dir = os.path.join(self.checkpointdir, "best")
        agent_dir = os.path.join(best_dir, "agent")
        os.makedirs(agent_dir, exist_ok=True)
        torch.save(
            deserialize_from_buffer(self.best_agent_group_params),
            os.path.join(agent_dir, "agent.pth"),
        )
        critic_dir = os.path.join(best_dir, "critic")
        os.makedirs(critic_dir, exist_ok=True)
        torch.save(
            deserialize_from_buffer(self.best_critic_params),
            os.path.join(critic_dir, "critic.pth"),
        )

    def _create_worker_group(self):
        """Create a MAPPOWorkerGroup for multi-GPU training.

        Returns ``None`` when ``use_multi_gpu`` is ``False`` (single-GPU mode).
        """
        if not self.use_multi_gpu:
            return None

        if self.experimental_message_target_kl is not None:
            raise ValueError("Experimental message KL requires single-device MAPPO")
        return MAPPOWorkerGroup(
            device_ids=self._get_device_ids(),
            agent_group_config=self.agent_group_config,
            critic_config=self.critic_config,
            critic_optimizer_config=self.critic_optimizer_config,
            agent_optimizer_config=self.agent_optimizer_config,
            gamma=self.gamma,
            clip_epsilon=self.clip_epsilon,
            gae_lambda=self.gae_lambda,
            entropy_coef=self.entropy_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
        )

    # ------------------------------------------------------------------
    # PPO learning (single- and multi-GPU)
    # ------------------------------------------------------------------

    def learn(
        self, sample_size, batch_size: int, times: int = 4
    ) -> dict[str, float]:
        """Run PPO updates on data sampled from the replay buffer.

        Parameters
        ----------
        sample_size : int
            Number of transitions to sample from the replay buffer.
        batch_size : int
            Mini-batch size for gradient computation.
        times : int
            Number of PPO epochs per learning call.

        Returns
        -------
        dict[str, float]
            Averaged total, actor, and critic losses.
        """
        if not self.use_multi_gpu:
            return self._learn_single_gpu(sample_size, batch_size, times)
        return self._learn_multi_gpu(sample_size, batch_size, times)

    @torch.no_grad()
    def _message_audit_forward(self, batch, *, actor=None, device=None, contiguous=False):
        """Recompute hard codes and masked policies without running training losses."""
        actor = self.eval_agent_group if actor is None else actor
        device = self.train_device if device is None else device
        observations = batch["observations"].to(device, dtype=torch.float32).transpose(1, 2)
        if contiguous:
            observations = observations.contiguous()
        alive = batch["alive_mask"][:, -1].to(device, dtype=torch.bool)
        groups = batch["group_indices"][:, -1].to(device, dtype=torch.long)
        padding = batch["timestep_padding_mask"].to(device, dtype=torch.bool)
        padding = padding[:, None, :].expand(-1, alive.shape[1], -1)
        result = actor(observations, padding, alive, groups)
        available = batch["avail_actions"]
        dist = masked_categorical(
            result["action_logits"],
            available[:, -1] if isinstance(available, torch.Tensor) else None,
            active_mask=alive,
        )
        return dist, result["message_codes"], alive

    @torch.no_grad()
    def _replay_message_audit(self, batch, actor_state):
        """Recreate rollout's actor, device and single-team input layout."""
        with torch.random.fork_rng(devices=[]):
            actor = self.agent_group_config.get_agent_group()
        actor.load_state_dict(actor_state, strict=True)
        device = self._message_rollout_device
        actor.eval().to(device)
        logits, codes = [], []
        for index in range(len(batch["observations"])):
            window = {key: None if value is None else value[index:index + 1]
                      for key, value in batch.items()}
            dist, code, _ = self._message_audit_forward(
                window, actor=actor, device=device, contiguous=True,
            )
            logits.append(dist.logits.cpu())
            codes.append(code.cpu())
        return torch.cat(logits), torch.cat(codes)

    @staticmethod
    def _message_audit_error(codes, log_probs, batch, alive):
        if not torch.equal(codes[alive], batch["message_codes"][:, -1].to(codes.device)[alive]):
            return "hard codes do not reproduce PPO rollout codes"
        recorded = batch["all_log_probs"][:, -1].to(log_probs)
        actual = log_probs[alive]
        expected = recorded[alive]
        if not (
            torch.isfinite(actual).all()
            and torch.isfinite(expected).all()
            and torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
        ):
            return "action probabilities do not reproduce PPO rollout log probabilities"
        return None

    def _save_message_audit(self, evidence):
        root = Path(self.workdir) / "message_audit"
        root.mkdir(parents=True, exist_ok=True)
        directory = Path(mkdtemp(prefix=f"round_{self.current_epoch + 1:04d}_", dir=root))
        path = directory / "reproduction.pt"
        pending = path.with_suffix(".tmp")
        torch.save(evidence, pending)
        pending.replace(path)
        return path

    def _capture_message_audit(self, batch, sample_ids=()):
        dist, codes, alive = self._message_audit_forward(batch)
        if not alive.any():
            raise ValueError("Message KL audit batch has no live agents")
        actions = batch["actions"][:, -1].to(device=alive.device, dtype=torch.long)
        log_probs = dist.log_prob(actions)
        mismatch = self._message_audit_error(codes, log_probs, batch, alive)
        if mismatch is None:
            return dist, codes, {}

        keys = (
            "observations", "timestep_padding_mask", "alive_mask", "group_indices",
            "avail_actions", "actions", "all_log_probs", "message_codes",
        )
        saved_batch = {
            key: batch[key].detach().cpu() if isinstance(batch[key], torch.Tensor) else None
            for key in keys
        }
        actor_state = {
            key: value.detach().cpu().clone()
            for key, value in get_state_dict(self.eval_agent_group).items()
        }
        evidence = {
            "round": self.current_epoch + 1,
            "sample_ids": [(int(episode), int(step)) for episode, step in sample_ids],
            "batch": saved_batch,
            "actor_state": actor_state,
            "agent_config": {
                "type": self.agent_group_config.ag_type,
                **self.agent_group_config.agent_group_config,
            },
            "training_device": str(self.train_device),
            "rollout_device": self._message_rollout_device,
            "torch_version": str(torch.__version__),
            "hip_version": torch.version.hip,
            "cuda_version": torch.version.cuda,
            "torch_threads": torch.get_num_threads(),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "training_mismatch": mismatch,
            "training_codes": codes.cpu(),
            "training_logits": dist.logits.cpu(),
            "training_action_log_probs": log_probs.cpu(),
        }

        metrics = {}
        try:
            replay_logits, replay_codes = self._replay_message_audit(saved_batch, actor_state)
            # Categorical.logits are already normalized; gather without renormalizing.
            replay_log_probs = replay_logits.gather(-1, actions.cpu().unsqueeze(-1)).squeeze(-1)
            error = self._message_audit_error(
                replay_codes, replay_log_probs, saved_batch, alive.cpu(),
            )
            evidence.update(replay_logits=replay_logits, replay_codes=replay_codes,
                            replay_action_log_probs=replay_log_probs)
            if error is None:
                behavior = torch.distributions.Categorical(logits=replay_logits.to(dist.logits))
                drift = torch.distributions.kl_divergence(behavior, dist)[alive]
                if not torch.isfinite(drift).all():
                    error = "non-finite policy drift between rollout and training forwards"
                else:
                    metrics = {
                        "message_behavior_replay_verified": 1.0,
                        "message_backend_code_mismatches": float(
                            (codes[alive] != replay_codes.to(codes.device)[alive]).sum()
                        ),
                        "message_backend_kl_mean": float(drift.mean()),
                        "message_backend_kl_max": float(drift.max()),
                        "message_backend_action_log_prob_max_abs": float(
                            (log_probs - replay_log_probs.to(log_probs))[alive].abs().max()
                        ),
                    }
        except Exception as exc:
            error = f"rollout replay failed: {type(exc).__name__}: {exc}"

        evidence.update(verified_rollout_replay=error is None, replay_error=error, metrics=metrics)
        path = self._save_message_audit(evidence)
        if error is not None:
            raise RuntimeError(f"Message audit: {error}. Evidence: {path}")
        logging.warning("Message audit: rollout replay verified; backend drift saved to %s", path)
        # PPO uses the logged behavior probabilities. KL early-stop compares
        # training-device forwards before and after updates, not two backends.
        return dist, codes, metrics

    def _learn_single_gpu(
        self, sample_size, batch_size: int, times: int = 4
    ) -> dict[str, float]:
        """Single-GPU PPO learning loop.

        For each PPO epoch (``times``), the full sampled dataset is
        iterated over in mini-batches.  Each batch runs:
          - forward pass through the critic (V(s))
          - forward pass through the agent (action logits)
          - fixed pre-update advantage/value targets
          - PPO clipped surrogate + entropy bonus  (actor)
          - MSE value loss  (critic)
          - separate backward passes with gradient clipping and optimizer steps.
        """
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_batches = 0

        self.eval_agent_group.to(self.train_device)
        self.eval_critic.to(self.train_device)

        dataset = self._prepare_ppo_dataset(sample_size, batch_size)
        audit_batch = None
        message_metrics = {}
        clip_count, clip_total = 0, 0
        if self.experimental_message_target_kl is not None and len(dataset):
            audit_loader = TrajectoryDataLoader(
                dataset, batch_size=batch_size, shuffle=False, num_workers=0
            )
            audit_batch = next(iter(audit_loader))
            sample_ids = getattr(dataset, "sample_id_list", ())[:batch_size]
            old_dist, old_codes, replay_metrics = self._capture_message_audit(audit_batch, sample_ids)
            message_metrics.update(replay_metrics)

        for epoch in range(times):
            dataloader = TrajectoryDataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.n_workers,
            )
            with tqdm(
                total=sample_size, desc=f"Times {epoch + 1}/{times}", unit="batch"
            ) as pbar:
                for batch in dataloader:
                    alive_mask = batch["alive_mask"].to(dtype=torch.bool)
                    observations = batch["observations"].to(dtype=torch.float32)
                    timestep_padding_mask = batch["timestep_padding_mask"].to(
                        dtype=torch.bool, device=self.train_device
                    )
                    states = batch["states"].to(dtype=torch.float32)
                    actions = batch["actions"].to(dtype=torch.int)
                    rewards = batch["rewards"].to(dtype=torch.float32)
                    all_log_probs = batch["all_log_probs"].to(dtype=torch.float32)

                    bs = states.shape[0]
                    n_agents = rewards.shape[2]

                    device = self.train_device
                    alive_mask = alive_mask.to(device)
                    states_dev = states.to(device)


                    timestep_padding_mask_expanded = torch.stack(
                        [timestep_padding_mask] * n_agents, dim=1
                    ).to(device)

                    # ---- Critic forward: full sequence -> V(s_{T-1}) only ----
                    self.eval_critic.train()
                    v = self.eval_critic(states_dev, alive_mask, timestep_padding_mask)["v"]
                    v_last = v[:, 0]  # (B,) — value at the last timestep of the segment

                    # Fixed pre-update advantages and value targets.
                    advantages_last, returns = ppo_targets(
                        batch, v_last, self.eval_critic, self.gamma, self._aggregate_rewards
                    )

                    # ---- Actor forward: action logits for last timestep ----
                    observations_transposed = torch.transpose(observations, 1, 2).to(
                        device
                    )
                    self.eval_agent_group.train()
                    ret_agent = agent_training_forward(
                        self.eval_agent_group,
                        observations_transposed,
                        timestep_padding_mask_expanded,
                        alive_mask[:,
                        -1,
                        :],
                        batch,
                        next_state=False,
                    )
                    action_logits = ret_agent["action_logits"]  # (B, N, action_dim)

                    actions_last = actions[:, -1].to(dtype=torch.int64, device=device)
                    log_probs_old = all_log_probs[:, -1, :].to(device)

                    # ---- PPO actor loss ----
                    avail_actions = batch["avail_actions"]
                    dist = masked_categorical(
                        action_logits,
                        avail_actions[:, -1] if isinstance(avail_actions, torch.Tensor) else None,
                        active_mask=alive_mask[:, -1],
                        actions=actions_last,
                    )
                    new_log_probs = dist.log_prob(actions_last)
                    entropy = dist.entropy()

                    alive_last_flag = alive_mask[:, -1, :].to(
                        dtype=torch.float32, device=device
                    )
                    alive_last_count = alive_last_flag.sum()

                    ratio = torch.exp(new_log_probs - log_probs_old)
                    adv_expanded = advantages_last.detach().unsqueeze(-1).expand(
                        -1, n_agents
                    )  # (B, N)
                    surr1 = ratio * adv_expanded
                    surr2 = (
                        torch.clamp(
                            ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
                        )
                        * adv_expanded
                    )
                    actor_loss = (
                        -(torch.min(surr1, surr2) * alive_last_flag).sum()
                        / max(alive_last_count, torch.tensor(1.0, device=device))
                    )
                    entropy_loss = (
                        -(entropy * alive_last_flag).sum()
                        / max(alive_last_count, torch.tensor(1.0, device=device))
                    )
                    actor_loss = actor_loss + self.entropy_coef * entropy_loss
                    if audit_batch is not None:
                        clipped = (ratio.detach() - 1).abs() > self.clip_epsilon
                        clip_count += int((clipped & alive_mask[:, -1]).sum())
                        clip_total += int(alive_last_count)
                        if total_batches == 0:
                            cosine = message_gradient_cosine(
                                actor_loss, ret_agent.get("relation_auxiliary_loss"),
                                self.eval_agent_group.codec,
                            )
                            if cosine is not None:
                                message_metrics["message_rl_relation_cosine"] = cosine
                    actor_loss = actor_loss + ret_agent.get("auxiliary_loss", 0.0)

                    if self.actor_auxiliary_objective is not None:
                        extra_loss = self.actor_auxiliary_objective(self.eval_agent_group)
                        if extra_loss.ndim != 0 or not torch.isfinite(extra_loss):
                            raise ValueError("Actor auxiliary objective must return a finite scalar")
                        actor_loss = actor_loss + extra_loss

                    # ---- Critic value loss (single timestep) ----
                    critic_loss = F.mse_loss(v_last, returns.detach())

                    # ---- Backward pass: actor ----
                    self.agent_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.eval_agent_group.parameters(), max_norm=self.max_grad_norm
                    )

                    # ---- Backward pass: critic ----
                    self.critic_optimizer.zero_grad()
                    (self.vf_coef * critic_loss).backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.eval_critic.parameters(), max_norm=self.max_grad_norm
                    )

                    self.agent_optimizer.step()
                    advance_communication_step(self.eval_agent_group)
                    # A disabled value objective must not update via optimizer
                    # momentum or weight decay either.
                    if self.vf_coef != 0:
                        self.critic_optimizer.step()

                    total_actor_loss += actor_loss.detach().cpu().item()
                    total_critic_loss += critic_loss.detach().cpu().item()
                    total_batches += 1

                    bs = batch["states"].shape[0]
                    pbar.update(bs)

            if audit_batch is not None:
                dist, codes, alive = self._message_audit_forward(audit_batch)
                kl = float(torch.distributions.kl_divergence(old_dist, dist)[alive].mean())
                if not math.isfinite(kl):
                    raise RuntimeError("Non-finite PPO hard-message policy KL")
                exceeded = kl > self.experimental_message_target_kl
                message_metrics.update({
                    "message_policy_kl": kl,
                    "message_clip_fraction": clip_count / max(clip_total, 1),
                    "message_code_flip_fraction": float(
                        (old_codes[alive] != codes[alive]).float().mean()
                    ) if codes.shape[-1] else 0.0,
                    "message_ppo_passes": epoch + 1,
                    "message_kl_threshold_exceeded": float(exceeded),
                })
                if exceeded:
                    break

        if message_metrics:
            logging.info("Experimental hard-message audit: %s", message_metrics)
            audit = getattr(self, "rolloutmanager_config", None)
            if hasattr(audit, "record_diagnostics"):
                audit.record_diagnostics(self.current_epoch, message_metrics)
        self.eval_agent_group.to("cpu")
        self.eval_critic.to("cpu")
        torch.cuda.empty_cache()

        avg_actor = total_actor_loss / max(total_batches, 1)
        avg_critic = total_critic_loss / max(total_batches, 1)
        return {
            "loss": avg_actor + avg_critic * self.vf_coef,
            "actor_loss": avg_actor,
            "critic_loss": avg_critic,
            **message_metrics,
        }

    def _learn_multi_gpu(
        self, sample_size, batch_size: int, times: int = 4
    ) -> dict[str, float]:
        """Multi-GPU PPO learning via worker processes.

        Each worker holds a full copy of the eval models and optimizers.
        Batches are sliced across workers, gradients are synchronised via
        all_reduce, and the average combined loss is collected.

        For each PPO epoch (``times``), a progress bar shows batch progress.
        """
        total_combined = 0.0
        total_batches = 0

        dataset = self._prepare_ppo_dataset(sample_size, batch_size)
        for epoch in range(times):
            dataloader = TrajectoryDataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.n_workers,
            )
            with tqdm(
                total=sample_size, desc=f"Times {epoch + 1}/{times}", unit="batch"
            ) as pbar:
                for batch in dataloader:
                    result = self.worker_group.train_step(batch)
                    total_combined += result["loss"]
                    total_batches += 1

                    bs = batch["states"].shape[0]
                    pbar.update(bs)

        return {"loss": total_combined / max(total_batches, 1)}

    # ------------------------------------------------------------------
    # On-policy training loop
    # ------------------------------------------------------------------

    def train(
        self,
        iterations,
        target_first_metric,
        batch_size=64,
        learning_times_per_iteration=1,
    ):
        """Run the on-policy MAPPO training loop.

        The loop follows this on-policy pattern::

            1. Initial ``evaluate()`` fills the replay buffer with episodes.
            2. For each iteration:
               a. ``learn()`` performs PPO updates on the current buffer.
               b. The buffer is cleared (on-policy: each batch is used once).
               c. ``evaluate()`` generates fresh episodes with the updated
                  policy and adds them to the buffer for the next iteration.
               d. LR schedulers are stepped; best model is tracked.

        Parameters
        ----------
        iterations : int
            Number of training iterations.
        target_first_metric : float
            Target value for the first eval metric (early stopping threshold).
        batch_size : int
            Mini-batch size for PPO updates.
        learning_times_per_iteration : int
            Number of PPO epochs per iteration.

        Returns
        -------
        dict
            Best metrics achieved during training.
        """
        self.eval_episodes_to_replay_ratio = 1.0

        # Initial data collection
        self.evaluate()

        for iteration in range(iterations):
            self.current_epoch = iteration

            sample_size = len(self.replaybuffer.buffer)
            if self.sample_mode == "ratio":
                sample_ratio = self.sample_ratio.get_value(iteration)
                sample_size = round(sample_size * sample_ratio)
            else:
                sample_size = round(self.sample_ratio.get_value(iteration))
            sample_size = min(sample_size, len(self.replaybuffer.buffer))
            if sample_size > 0:
                agent_group_lr = self.agent_optimizer.param_groups[0]["lr"]
                critic_lr = self.critic_optimizer.param_groups[0]["lr"]
                logging.info(
                    f"Iteration {iteration}: Batch size: {batch_size}, "
                    f"Critic lr: {critic_lr:.8f}, Agent lr: {agent_group_lr:.8f}"
                )
                self._sync_params_to_workers()
                train_result = self.learn(
                    sample_size=sample_size,
                    batch_size=batch_size,
                    times=learning_times_per_iteration,
                )
                if self.worker_group is not None:
                    self.worker_group.average_eval_params()
                self._sync_eval_params_from_workers()
                logging.info(
                    f"Iteration {iteration}: Loss {train_result['loss']:.4f}"
                )

            # Clear buffer (on-policy: discard old data)
            self.replaybuffer = self.replaybuffer_config.create_replaybuffer()

            # A bounded experiment learns from the last collected batch, then stops.
            if getattr(self.rolloutmanager_config, "step_budget", None) is not None and (
                self.rolloutmanager_config.collection_exhausted or iteration + 1 == iterations
            ):
                break

            # Collect fresh episodes with updated policy
            result = self.evaluate()
            metrics = {key: result[key]["mean"] for key in self.eval_metric_list}
            first_metric = next(iter(metrics.values()))
            first_metric_name = next(iter(metrics.keys()))
            self.save_intermediate_results(iteration, result)

            if isinstance(
                self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.lr_scheduler.step(first_metric)
            elif isinstance(self.lr_scheduler, torch.optim.lr_scheduler.LRScheduler):
                self.lr_scheduler.step()

            if isinstance(
                self.agent_lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.agent_lr_scheduler.step(first_metric)
            elif isinstance(
                self.agent_lr_scheduler, torch.optim.lr_scheduler.LRScheduler
            ):
                self.agent_lr_scheduler.step()

            if first_metric >= self.best_metrics.get(first_metric_name, -np.inf):
                self.best_metrics = metrics
                self.best_agent_group_params = serialize_to_buffer(
                    get_state_dict(self.eval_agent_group)
                )
                self.best_critic_params = serialize_to_buffer(
                    get_state_dict(self.eval_critic)
                )

            if first_metric >= target_first_metric:
                break

        logging.info(
            f"Best strategy: {yaml.dump(self.best_metrics, default_flow_style=False, sort_keys=False)}"
        )
        if getattr(self.rolloutmanager_config, "step_budget", None) is None or any(
            np.isfinite(metric) for metric in self.best_metrics.values()
        ):
            self.save_best_model()
        return self.best_metrics
