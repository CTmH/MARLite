"""SelfSupervisedMAPPOTrainer — MAPPO base class with SSL infrastructure.

Mirrors ``SelfSupervisedQMIXTrainer`` for the on-policy MAPPO family.
All SSL components (model, optimiser, data constructor, checkpoint
handling) are **required** and initialised unconditionally — subclasses
(``SSLGroupConsensusMAPPOTrainer``) only add SSL-specific logic.
"""

import os
import yaml
import time
import numpy as np
import torch
from absl import logging
from torch.nn.modules.loss import _Loss

from marlite.algorithm.model import ModelConfig
from marlite.trainer.onpolicy_trainer import OnPolicyTrainer
from marlite.algorithm.critic.mixer import Mixer as MixerCritic
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


class SelfSupervisedMAPPOTrainer(OnPolicyTrainer):
    """MAPPO trainer with mandatory SSL (self-supervised learning) support.

    Every subclass of this trainer **must** configure SSL.  The SSL model,
    optimiser, data constructor, and reconstruction loss are created
    unconditionally — no ``is None`` guards are needed downstream.

    Provides:
    - SSL model / optimiser / LR scheduler / data constructor lifecycle.
    - ``_compute_ssl_loss`` and ``_combine_rl_ssl_loss`` helpers.
    - Checkpoint save / load extended to include the SSL model.
    - Worker parameter sync extended to include the SSL model.
    - SSL LR logging and scheduler stepping in ``train()``.

    Parameters
    ----------
    ssl_model_config : ModelConfig
        Configuration for the SSL decoder / reconstruction model.
    ssl_optimizer_config : OptimizerConfig
        Optimizer configuration for the SSL model.
    ssl_lr_scheduler_conf : LRSchedulerConfig or None
        LR scheduler configuration for the SSL optimizer.
    data_constructor_config : SelfSupervisedDataConstructorConfig
        Configuration for the data constructor that builds reconstruction
        targets.
    reconstruction_loss : _Loss
        Loss function for reconstruction (e.g., ``PointSetMSELoss``).
    self_supervised_learning_loss_weight : float
        Weight ``w_ssl`` for VAE loss in ``weighted_sum`` mode.
    loss_combination_method : str
        ``"weighted_sum"`` or ``"pit_loss"``.
    pit_loss_alpha : float
        Alpha parameter for ``PITLoss``.
    clip_epsilon : float
        PPO clip range for the importance sampling ratio.
    gae_lambda : float
        GAE lambda parameter controlling bias-variance tradeoff.
    entropy_coef : float
        Coefficient for the entropy bonus.
    vf_coef : float
        Coefficient for the value function loss.
    max_grad_norm : float
        Maximum gradient norm for clipping.
    **kwargs
        Forwarded to ``OnPolicyTrainer.__init__``.
    """

    def __init__(
        self,
        # SSL params (required)
        ssl_model_config: ModelConfig,
        ssl_optimizer_config: OptimizerConfig,
        ssl_lr_scheduler_conf: LRSchedulerConfig | None,
        data_constructor_config: SelfSupervisedDataConstructorConfig,
        reconstruction_loss: _Loss,
        self_supervised_learning_loss_weight: float = 1.0,
        loss_combination_method: str = "weighted_sum",
        pit_loss_alpha: float = 0.9,
        # PPO params
        clip_epsilon: float = 0.2,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 5.0,
        **kwargs,
    ):
        # -- PPO params (must be set before super().__init__) ------------
        self.clip_epsilon = clip_epsilon
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        # -- SSL configuration ---------------------------------------------
        self.ssl_model_config = ssl_model_config
        self.ssl_optimizer_config = ssl_optimizer_config
        self.ssl_lr_scheduler_conf = ssl_lr_scheduler_conf
        self.data_constructor_config = data_constructor_config
        self.reconstruction_loss = reconstruction_loss
        if not isinstance(self.reconstruction_loss, ReconstructionLoss):
            raise TypeError(
                f"reconstruction_loss must be a ReconstructionLoss subclass, "
                f"got {type(self.reconstruction_loss).__name__}"
            )
        self.self_supervised_learning_loss_weight = (
            self_supervised_learning_loss_weight
        )
        self.loss_combination_method = loss_combination_method
        self.pit_loss_alpha = pit_loss_alpha

        # -- Data constructor (always created) -----------------------------
        self.data_constructor = self.data_constructor_config.get_data_constructor()

        # -- SSL model & optimiser (must exist before super().__init__)
        #    so that _sync_params_to_workers (called during
        #    _setup_multi_gpu in OnPolicyTrainer.__init__) can access them.
        self.ssl_model = ssl_model_config.get_model()
        self.ssl_optimizer = self.ssl_optimizer_config.get_optimizer(
            self.ssl_model.parameters()
        )

        super().__init__(**kwargs)

        if isinstance(self.eval_critic, MixerCritic):
            raise TypeError(
                "Critic subclass required, not Mixer subclass"
            )

        # -- SSL LR scheduler ----------------------------------------------
        if isinstance(ssl_lr_scheduler_conf, LRSchedulerConfig):
            self.ssl_lr_scheduler = ssl_lr_scheduler_conf.get_lr_scheduler(
                self.ssl_optimizer
            )
        else:
            self.ssl_lr_scheduler = None

        # -- Optional torch.compile for ssl_model (single-GPU only) --------
        if self.compile_models:
            self.ssl_model = torch.compile(
                self.ssl_model.to(self.train_device)
            ).to("cpu")

        # -- PITLoss for combining RL + SSL losses -------------------------
        self.pit_loss = PITLoss(
            num_tasks=2, alpha=self.pit_loss_alpha, reduction="mean"
        )

        # -- Checkpoint caches ---------------------------------------------
        self.best_agent_group_params = serialize_to_buffer(
            get_state_dict(self.eval_agent_group)
        )
        self.best_critic_params = serialize_to_buffer(
            get_state_dict(self.eval_critic)
        )
        self.best_ssl_model_params = serialize_to_buffer(
            self.ssl_model.state_dict()
        )
        self._cached_ssl_model_params = serialize_to_buffer(
            self.ssl_model.state_dict()
        )

    # ------------------------------------------------------------------
    # SSL helpers
    # ------------------------------------------------------------------

    def _compute_ssl_loss(self, pred_set, target_set, mask=None):
        """Compute reconstruction loss, respecting ``ReconstructionLoss`` API."""
        return self.reconstruction_loss(pred_set, target_set, mask)

    def _combine_rl_ssl_loss(self, rl_loss, ssl_loss):
        """Combine RL and SSL losses via weighted sum or PITLoss."""
        if self.loss_combination_method == "pit_loss":
            losses = torch.stack([rl_loss, ssl_loss])
            return self.pit_loss(losses)
        return rl_loss + self.self_supervised_learning_loss_weight * ssl_loss

    # ------------------------------------------------------------------
    # PPO learning dispatch (single- / multi-GPU)
    # ------------------------------------------------------------------

    def learn(self, sample_size, batch_size: int, times: int = 4):
        if not self.use_multi_gpu:
            return self._learn_single_gpu(sample_size, batch_size, times)
        return self._learn_multi_gpu(sample_size, batch_size, times)

    # ------------------------------------------------------------------
    # Multi-GPU parameter sync (extended with ssl_model)
    # ------------------------------------------------------------------

    def _sync_params_to_workers(self):
        if self.worker_group is None:
            return
        trainable_params = {
            "eval_agent_group": get_state_dict(self.eval_agent_group),
            "eval_critic": get_state_dict(self.eval_critic),
            "ssl_model": get_state_dict(self.ssl_model),
            "reward_aggr_mode": self.reward_aggr_mode,
        }
        self.worker_group.broadcast_params(trainable_params)
        critic_lr = self.critic_optimizer.param_groups[0]["lr"]
        agent_lr = self.agent_optimizer.param_groups[0]["lr"]
        self.worker_group.sync_lr_to_workers(critic_lr, agent_lr)

    def _sync_eval_params_from_workers(self):
        if self.worker_group is None:
            return
        eval_params = self.worker_group.read_params_from_worker0()
        load_state_dict_into(self.eval_agent_group, eval_params["eval_agent_group"])
        load_state_dict_into(self.eval_critic, eval_params["eval_critic"])
        load_state_dict_into(self.ssl_model, eval_params["ssl_model"])

    # ------------------------------------------------------------------
    # Checkpoint (includes ssl_model)
    # ------------------------------------------------------------------

    def save_current_model(self, checkpoint: str):
        super().save_current_model(checkpoint)
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

    def save_best_model(self):
        """Write cached best params (agent, critic, ssl) directly to disk."""
        import os
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
        ssl_dir = os.path.join(best_dir, "ssl_model")
        os.makedirs(ssl_dir, exist_ok=True)
        torch.save(
            deserialize_from_buffer(self.best_ssl_model_params),
            os.path.join(ssl_dir, "ssl_model.pth"),
        )

    # ------------------------------------------------------------------
    # On-policy training loop (extends MAPPO with SSL LR scheduler)
    # ------------------------------------------------------------------

    def train(
        self,
        iterations,
        target_first_metric,
        batch_size=64,
        learning_times_per_iteration=1,
    ):
        self.eval_episodes_to_replay_ratio = 1.0
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
                agent_lr = self.agent_optimizer.param_groups[0]["lr"]
                critic_lr = self.critic_optimizer.param_groups[0]["lr"]
                ssl_lr = self.ssl_optimizer.param_groups[0]["lr"]
                logging.info(
                    f"Iteration {iteration}: Batch size: {batch_size}, "
                    f"Critic lr: {critic_lr:.8f}, Agent lr: {agent_lr:.8f}, "
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
                self.best_agent_group_params = serialize_to_buffer(
                    get_state_dict(self.eval_agent_group)
                )
                self.best_critic_params = serialize_to_buffer(
                    get_state_dict(self.eval_critic)
                )
                self.best_ssl_model_params = serialize_to_buffer(
                    get_state_dict(self.ssl_model)
                )

            if first_metric >= target_first_metric:
                break

        logging.info(
            f"Best strategy: {yaml.dump(self.best_metrics, default_flow_style=False, sort_keys=False)}"
        )
        self.save_best_model()
        return self.best_metrics
