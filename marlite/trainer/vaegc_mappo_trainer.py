"""VAE Group Consensus MAPPO trainer.

Combines the on-policy MAPPO algorithm (PPO clipped surrogate, GAE,
centralised value critic) with VAE-based group consensus learning.
The consensus latent is trained via self-supervised reconstruction of
global state, while the PPO objectives drive the policy and value updates.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn.modules.loss import _Loss
from absl import logging

from marlite.algorithm.model import ModelConfig
from marlite.trainer.mappo_trainer import MAPPOTrainer
from marlite.trainer.onpolicy_trainer import OnPolicyTrainer
from marlite.util.serialization import (
    serialize_to_buffer,
    deserialize_from_buffer,
    get_state_dict,
    load_state_dict_into,
)
from marlite.util.optimizer_config import OptimizerConfig
from marlite.util.lr_scheduler_config import LRSchedulerConfig
from marlite.util.self_supervised_data_constructor.self_supervised_data_constructor_config import (
    SelfSupervisedDataConstructorConfig,
)
from marlite.util.loss_func import ReconstructionLoss, PITLoss
from marlite.util.trajectory_dataset import TrajectoryDataLoader


class VAEGroupConsensusMAPPOTrainer(MAPPOTrainer):
    """MAPPO trainer with VAE group consensus self-supervised learning.

    Extends :class:`MAPPOTrainer` with:
    - SSL decoder model for reconstructing global state from group consensus.
    - KL divergence regularisation on the latent distributions.
    - Optional warmup phase (PPO-only before enabling SSL).
    - Flexible loss combination ("weighted_sum" or "pit_loss").

    Parameters
    ----------
    ssl_model_config : ModelConfig
        Configuration for the SSL decoder model.
    ssl_optimizer_config : OptimizerConfig
        Optimizer configuration for the SSL model.
    ssl_lr_scheduler_conf : LRSchedulerConfig
        LR scheduler configuration for the SSL optimizer.
    data_constructor_config : SelfSupervisedDataConstructorConfig
        Configuration for the data constructor that builds reconstruction
        targets from global state and group assignments.
    reconstruction_loss : _Loss
        Loss function for reconstruction (e.g. PointSetMSELoss).
    self_supervised_learning_loss_weight : float
        Weight w_ssl for VAE loss in combined loss.
    loss_combination_method : str
        "weighted_sum" or "pit_loss".
    pit_loss_alpha : float
        Alpha parameter for PITLoss exponential decay.
    kl_divergence_weight : float
        Beta_KL weight for KL divergence term.
    recon_mode : str
        "per_agent" or "per_group" reconstruction mode.
    kl_on_agent : bool
        Apply KL divergence on per-agent latent distributions.
    kl_on_group : bool
        Apply KL divergence on per-group (deduplicated) distributions.
    warmup_iterations : int
        Number of initial iterations with PPO-only (no SSL).
    """

    def __init__(
        self,
        # MAPPO params (see MAPPOTrainer)
        clip_epsilon: float = 0.2,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 5.0,
        # SSL params
        ssl_model_config: ModelConfig = None,
        ssl_optimizer_config: OptimizerConfig = None,
        ssl_lr_scheduler_conf: LRSchedulerConfig = None,
        data_constructor_config: SelfSupervisedDataConstructorConfig = None,
        reconstruction_loss: _Loss = None,
        self_supervised_learning_loss_weight: float = 1.0,
        loss_combination_method: str = "weighted_sum",
        pit_loss_alpha: float = 0.9,
        # VAE params
        kl_divergence_weight: float = 0.005,
        recon_mode: str = "per_agent",
        kl_on_agent: bool = True,
        kl_on_group: bool = False,
        warmup_iterations: int = 0,
        **kwargs,
    ):
        if recon_mode not in ("per_agent", "per_group"):
            raise ValueError(
                f"recon_mode must be 'per_agent' or 'per_group', got '{recon_mode}'"
            )
        self.recon_mode = recon_mode
        self.kl_divergence_weight = kl_divergence_weight
        self.kl_on_agent = kl_on_agent
        self.kl_on_group = kl_on_group
        self.warmup_iterations = warmup_iterations
        self.self_supervised_learning_loss_weight = (
            self_supervised_learning_loss_weight
        )
        self.loss_combination_method = loss_combination_method
        self.pit_loss_alpha = pit_loss_alpha
        self.reconstruction_loss = reconstruction_loss

        if data_constructor_config is not None:
            self.data_constructor = data_constructor_config.get_data_constructor()
        else:
            self.data_constructor = None

        super().__init__(
            clip_epsilon=clip_epsilon,
            gae_lambda=gae_lambda,
            entropy_coef=entropy_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            **kwargs,
        )

        if ssl_model_config is not None:
            self.ssl_model = ssl_model_config.get_model()
        else:
            self.ssl_model = None

        if ssl_optimizer_config is not None and self.ssl_model is not None:
            self.ssl_optimizer = ssl_optimizer_config.get_optimizer(
                self.ssl_model.parameters()
            )
        else:
            self.ssl_optimizer = None

        if (
            ssl_lr_scheduler_conf is not None
            and self.ssl_optimizer is not None
        ):
            self.ssl_lr_scheduler = ssl_lr_scheduler_conf.get_lr_scheduler(
                self.ssl_optimizer
            )
        else:
            self.ssl_lr_scheduler = None

        if self.compile_models and self.ssl_model is not None:
            self.ssl_model = torch.compile(
                self.ssl_model.to(self.train_device)
            ).to("cpu")

        self.pit_loss = PITLoss(
            num_tasks=2, alpha=self.pit_loss_alpha, reduction="mean"
        )

    # ------------------------------------------------------------------
    # SSL helper methods
    # ------------------------------------------------------------------

    def _compute_ssl_loss(self, pred_set, target_set, mask=None):
        if isinstance(self.reconstruction_loss, ReconstructionLoss):
            return self.reconstruction_loss(pred_set, target_set, mask)
        return self.reconstruction_loss(pred_set, target_set)

    def _combine_rl_ssl_loss(self, rl_loss, ssl_loss):
        if self.loss_combination_method == "pit_loss":
            losses = torch.stack([rl_loss, ssl_loss])
            return self.pit_loss(losses)
        return rl_loss + self.self_supervised_learning_loss_weight * ssl_loss

    def _build_recon_targets(self, observations, states, group_indices, alive_mask):
        obs_np = observations.detach().cpu().numpy()
        st_np = (
            states.detach().cpu().numpy()
            if isinstance(states, torch.Tensor)
            else states
        )
        alv_np = alive_mask.detach().cpu().numpy()

        targets_np, construct_mask_np = self.data_constructor.process(
            observations=obs_np,
            states=st_np,
            grouping=group_indices,
            alive_mask=alv_np,
        )
        targets = torch.tensor(
            targets_np, dtype=torch.float32, device=self.train_device
        )
        construct_mask = torch.tensor(
            construct_mask_np, dtype=torch.bool, device=self.train_device
        )
        return targets, construct_mask

    def _recon_loss_per_group(self, consensus, targets, construct_mask):
        bs, G, L = consensus.shape
        target_flat_dim = targets.shape[2:]
        pred_flat = self.ssl_model(consensus.reshape(bs * G, L))
        pred_g = pred_flat.reshape(bs, G, *target_flat_dim)
        return self._compute_ssl_loss(pred_g, targets, construct_mask)

    def _recon_loss_per_agent(
        self, consensus, group_indices, targets, construct_mask, alive_mask
    ):
        bs, G, L = consensus.shape
        n_agents = group_indices.shape[1]
        target_flat_dim = targets.shape[2:]

        pred_flat = self.ssl_model(consensus.reshape(bs * G, L))
        pred_g = pred_flat.reshape(bs, G, *target_flat_dim)

        gids = torch.as_tensor(group_indices, device=consensus.device)
        dead = gids < 0
        mask_oh = F.one_hot(gids.clamp(min=0), num_classes=G).float()
        mask_oh[dead] = 0.0
        mask_oh = mask_oh * construct_mask.unsqueeze(1).float()

        flat_pred = pred_g.reshape(bs, G, -1)
        flat_target = targets.reshape(bs, G, -1)
        pred_agent = torch.bmm(mask_oh, flat_pred).reshape(
            bs, n_agents, *target_flat_dim
        )
        target_agent = torch.bmm(mask_oh, flat_target).reshape(
            bs, n_agents, *target_flat_dim
        )
        agent_alive = alive_mask[:, -1, :]
        return self._compute_ssl_loss(pred_agent, target_agent, agent_alive)

    def _compute_kl_divergence(
        self, agent_mu, agent_log_var, alive_mask, group_mu, group_log_var, construct_mask
    ):
        kl = torch.tensor(0.0, device=self.train_device)
        if self.kl_on_agent:
            mask = alive_mask[:, -1, :].unsqueeze(-1).expand_as(agent_mu)
            kl_per_dim = (
                1 + agent_log_var - agent_mu.pow(2) - torch.exp(agent_log_var)
            )
            kl = kl + (-0.5 * (kl_per_dim * mask).sum() / mask.sum().clamp(min=1))
        if self.kl_on_group:
            mask = construct_mask.unsqueeze(-1).expand_as(group_mu)
            kl_per_dim = (
                1 + group_log_var - group_mu.pow(2) - torch.exp(group_log_var)
            )
            kl = kl + (-0.5 * (kl_per_dim * mask).sum() / mask.sum().clamp(min=1))
        return kl

    # ------------------------------------------------------------------
    # PPO + VAE learning
    # ------------------------------------------------------------------

    def _learn_single_gpu(self, sample_size, batch_size: int, times: int = 4):
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_vae_loss = 0.0
        total_batches = 0

        self.eval_agent_group.to(self.train_device)
        self.eval_critic.to(self.train_device)
        if self.ssl_model is not None:
            self.ssl_model.to(self.train_device)

        is_warmup = self.current_epoch < self.warmup_iterations

        dataset = self.replaybuffer.sample(sample_size)
        dataloader = TrajectoryDataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=self.n_workers
        )

        for epoch in range(times):
            for batch in dataloader:
                alive_mask = batch["alive_mask"].to(dtype=torch.bool)
                observations = batch["observations"].to(dtype=torch.float32)
                timestep_padding_mask = batch["timestep_padding_mask"].to(
                    dtype=torch.bool
                )
                states = batch["states"].to(dtype=torch.float32)
                actions = batch["actions"].to(dtype=torch.int)
                rewards = batch["rewards"].to(dtype=torch.float32)
                next_states = batch["next_states"].to(dtype=torch.float32)
                next_timestep_padding_mask = batch[
                    "next_timestep_padding_mask"
                ].to(dtype=torch.bool)
                next_alive_mask = batch["next_alive_mask"].to(dtype=torch.bool)
                all_log_probs = batch["all_log_probs"].to(dtype=torch.float32)
                terminations = batch["terminations"].to(dtype=torch.bool)

                bs = states.shape[0]
                n_agents = rewards.shape[2]
                t_steps = rewards.shape[1]
                device = self.train_device

                alive_mask_d = alive_mask.to(device)
                next_alive_mask_d = next_alive_mask.to(device)
                states_dev = states.to(device)
                next_states_dev = next_states.to(device)

                # ── Precompute group_indices from batch ──
                group_indices_batch = batch.get("group_indices")
                if group_indices_batch is not None:
                    group_indices_np = group_indices_batch[:, -1, :].numpy()
                else:
                    group_indices_np = np.zeros((bs, n_agents), dtype=np.int64) - 1

                # ── Critic forward: full sequence → V(s_{T-1}) only ──
                self.eval_critic.train()
                v = self.eval_critic(states_dev, alive_mask_d, timestep_padding_mask)["v"]
                v_last = v[:, 0]  # (B,)

                with torch.no_grad():
                    v_next = self.eval_critic(
                        next_states_dev[:, -1:, ...],
                        next_alive_mask_d[:, -1:, ...],
                        next_timestep_padding_mask[:, -1:],
                    )["v"][:, 0]  # (B,)

                # ── Single-step TD residual as advantage ──
                r_last = rewards.sum(dim=2)[:, -1].to(device)
                done_last = terminations.any(dim=2)[:, -1].to(
                    dtype=torch.float32, device=device
                )
                delta = r_last + self.gamma * v_next * (1.0 - done_last) - v_last
                advantages_last = delta  # (B,)
                returns = delta + v_last  # (B,)

                # ── Agent forward: action_logits + consensus ──
                states_last_np = states_dev[:, -1].detach().cpu().numpy()
                timestep_padding_mask_expanded = torch.stack(
                    [timestep_padding_mask] * n_agents, dim=1
                ).to(device)
                observations_transposed = torch.transpose(
                    observations, 1, 2
                ).to(device)

                self.eval_agent_group.reset().train()
                ret_agent = self.eval_agent_group(
                    observations_transposed,
                    states_last_np,
                    timestep_padding_mask_expanded,
                    alive_mask_d[:, -1, :],
                    group_indices_np,
                )
                action_logits = ret_agent["action_logits"]
                group_consensus = ret_agent.get("group_consensus")
                group_mu = ret_agent.get("group_mu")
                group_log_var = ret_agent.get("group_log_var")
                agent_mu = ret_agent.get("agent_mu")
                agent_log_var = ret_agent.get("agent_log_var")

                # ── PPO actor loss ──
                actions_last = actions[:, -1].to(
                    dtype=torch.int64, device=device
                )
                log_probs_old = all_log_probs[:, -1, :].to(device)

                dist = Categorical(logits=action_logits)
                new_log_probs = dist.log_prob(actions_last)
                entropy = dist.entropy()

                alive_last_flag = alive_mask_d[:, -1, :].to(
                    dtype=torch.float32, device=device
                )
                alive_last_count = alive_last_flag.sum()

                ratio = torch.exp(new_log_probs - log_probs_old)
                adv_expanded = advantages_last.unsqueeze(-1).expand(-1, n_agents)
                surr1 = ratio * adv_expanded
                surr2 = (
                    torch.clamp(
                        ratio,
                        1.0 - self.clip_epsilon,
                        1.0 + self.clip_epsilon,
                    )
                    * adv_expanded
                )
                actor_loss = (
                    -(torch.min(surr1, surr2) * alive_last_flag).sum()
                    / max(
                        alive_last_count,
                        torch.tensor(1.0, device=device),
                    )
                )
                entropy_loss = (
                    -(entropy * alive_last_flag).sum()
                    / max(
                        alive_last_count,
                        torch.tensor(1.0, device=device),
                    )
                )
                actor_loss = actor_loss + self.entropy_coef * entropy_loss

                # ── PPO critic loss ──
                critic_loss = F.mse_loss(v_last, returns.detach())

                # ── VAE reconstruction loss ──
                if (
                    is_warmup
                    or self.data_constructor is None
                    or self.ssl_model is None
                    or group_consensus is None
                ):
                    vae_loss = torch.tensor(0.0, device=device)
                else:
                    targets, construct_mask = self._build_recon_targets(
                        observations, states, group_indices_np, alive_mask
                    )
                    if self.recon_mode == "per_group":
                        recon_loss = self._recon_loss_per_group(
                            group_consensus, targets, construct_mask
                        )
                    else:
                        recon_loss = self._recon_loss_per_agent(
                            group_consensus,
                            group_indices_np,
                            targets,
                            construct_mask,
                            alive_mask_d,
                        )
                    kl = self._compute_kl_divergence(
                        agent_mu,
                        agent_log_var,
                        alive_mask_d,
                        group_mu,
                        group_log_var,
                        construct_mask,
                    )
                    vae_loss = recon_loss + self.kl_divergence_weight * kl

                # ── Combined loss ──
                rl_loss = actor_loss + self.vf_coef * critic_loss
                if is_warmup or vae_loss.item() == 0.0:
                    combined_loss = rl_loss
                else:
                    combined_loss = self._combine_rl_ssl_loss(
                        rl_loss, vae_loss
                    )

                # ── Backward ──
                self.agent_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                if self.ssl_optimizer is not None:
                    self.ssl_optimizer.zero_grad()

                combined_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.eval_agent_group.parameters(),
                    max_norm=self.max_grad_norm,
                )
                torch.nn.utils.clip_grad_norm_(
                    self.eval_critic.parameters(),
                    max_norm=self.max_grad_norm,
                )
                if self.ssl_model is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.ssl_model.parameters(),
                        max_norm=self.max_grad_norm,
                    )

                self.agent_optimizer.step()
                self.critic_optimizer.step()
                if self.ssl_optimizer is not None:
                    self.ssl_optimizer.step()

                total_actor_loss += actor_loss.detach().cpu().item()
                total_critic_loss += critic_loss.detach().cpu().item()
                total_vae_loss += (
                    vae_loss.detach().cpu().item()
                    if isinstance(vae_loss, torch.Tensor)
                    else vae_loss
                )
                total_batches += 1

        self.eval_agent_group.to("cpu")
        self.eval_critic.to("cpu")
        if self.ssl_model is not None:
            self.ssl_model.to("cpu")
        torch.cuda.empty_cache()

        avg_rl = (total_actor_loss + total_critic_loss * self.vf_coef) / max(
            total_batches, 1
        )
        avg_vae = total_vae_loss / max(total_batches, 1)
        logging.info(
            f"  Iter {self.current_epoch}: RL Loss {avg_rl:.4f}, VAE Loss {avg_vae:.4f}"
        )
        return avg_rl + avg_vae

    # ------------------------------------------------------------------
    # Checkpoint (includes ssl_model)
    # ------------------------------------------------------------------

    def save_current_model(self, checkpoint: str):
        super().save_current_model(checkpoint)
        if self.ssl_model is not None:
            ssl_path = os.path.join(
                self.checkpointdir, checkpoint, "ssl_model"
            )
            os.makedirs(ssl_path, exist_ok=True)
            self.ssl_model.to("cpu")
            torch.save(
                get_state_dict(self.ssl_model),
                os.path.join(ssl_path, "ssl_model.pth"),
            )
        return self

    def load_checkpoint(self, checkpoint: str):
        super().load_checkpoint(checkpoint)
        if self.ssl_model is not None:
            ssl_path = os.path.join(
                self.checkpointdir, checkpoint, "ssl_model", "ssl_model.pth"
            )
            if os.path.exists(ssl_path):
                self.ssl_model.to("cpu")
                load_state_dict_into(
                    self.ssl_model,
                    torch.load(ssl_path, weights_only=True),
                )
        return self

    # ------------------------------------------------------------------
    # On-policy training loop (extends MAPPO with SSL LR scheduler)
    # ------------------------------------------------------------------

    def train(
        self,
        iterations,
        target_first_metric,
        batch_size=64,
        learning_times_per_iteration=4,
    ):
        self.eval_episodes_to_replay_ratio = 1.0
        self.evaluate()

        for iteration in range(iterations):
            self.current_epoch = iteration

            sample_size = len(self.replaybuffer.buffer)
            if sample_size > 0:
                agent_group_lr = self.agent_optimizer.param_groups[0]["lr"]
                critic_lr = self.critic_optimizer.param_groups[0]["lr"]
                ssl_lr = (
                    self.ssl_optimizer.param_groups[0]["lr"]
                    if self.ssl_optimizer is not None
                    else 0.0
                )
                logging.info(
                    f"Iteration {iteration}: Batch size: {batch_size}, "
                    f"Critic lr: {critic_lr:.8f}, Agent lr: {agent_group_lr:.8f}, "
                    f"SSL lr: {ssl_lr:.8f}"
                )
                self._sync_params_to_workers()
                loss = self.learn(
                    sample_size=sample_size,
                    batch_size=batch_size,
                    times=learning_times_per_iteration,
                )
                self._sync_eval_params_from_workers()
                logging.info(f"Iteration {iteration}: Loss {loss:.4f}")

            self.replaybuffer = self.replaybuffer_config.create_replaybuffer()
            result = self.evaluate()
            metrics = {
                key: result[key]["mean"] for key in self.eval_metric_list
            }
            first_metric = next(iter(metrics.values()))
            first_metric_name = next(iter(metrics.keys()))
            self.save_intermediate_results(iteration, result)

            if isinstance(
                self.lr_scheduler,
                torch.optim.lr_scheduler.ReduceLROnPlateau,
            ):
                self.lr_scheduler.step(first_metric)
            elif isinstance(
                self.lr_scheduler, torch.optim.lr_scheduler.LRScheduler
            ):
                self.lr_scheduler.step()

            if isinstance(
                self.agent_lr_scheduler,
                torch.optim.lr_scheduler.ReduceLROnPlateau,
            ):
                self.agent_lr_scheduler.step(first_metric)
            elif isinstance(
                self.agent_lr_scheduler,
                torch.optim.lr_scheduler.LRScheduler,
            ):
                self.agent_lr_scheduler.step()

            if self.ssl_lr_scheduler is not None:
                if isinstance(
                    self.ssl_lr_scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    self.ssl_lr_scheduler.step(first_metric)
                elif isinstance(
                    self.ssl_lr_scheduler,
                    torch.optim.lr_scheduler.LRScheduler,
                ):
                    self.ssl_lr_scheduler.step()

            if first_metric >= self.best_metrics.get(
                first_metric_name, -np.inf
            ):
                self.best_metrics = metrics
                self.save_current_model(checkpoint="best")

            if first_metric >= target_first_metric:
                break

        return self.best_metrics
