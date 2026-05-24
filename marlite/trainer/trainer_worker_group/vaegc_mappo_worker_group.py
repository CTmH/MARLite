"""
VAE Group Consensus MAPPO worker group for multi-GPU training.
"""

from typing import Any, Dict
from marlite.trainer.trainer_worker_group.base_worker_group import (
    BaseWorkerGroup,
    _slice_batch,
)


class VAEGroupConsensusMAPPOWorkerGroup(OnPolicyWorkerGroup):
    def __init__(
        self,
        device_ids: list,
        agent_group_config,
        critic_config,
        critic_optimizer_config,
        agent_optimizer_config,
        gamma: float = 0.99,
        max_grad_norm: float = 5.0,
        clip_epsilon: float = 0.2,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        vf_coef: float = 0.5,
        ssl_model_config=None,
        ssl_optimizer_config=None,
        reconstruction_loss=None,
        data_constructor=None,
        kl_divergence_weight: float = 0.005,
        self_supervised_learning_loss_weight: float = 1.0,
        loss_combination_method: str = "weighted_sum",
        pit_loss_alpha: float = 0.9,
        warmup_iterations: int = 0,
        recon_mode: str = "per_agent",
        kl_on_agent: bool = True,
        kl_on_group: bool = False,
        init_method: str = None,
    ):
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.clip_epsilon = clip_epsilon
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.agent_group_config = agent_group_config
        self.critic_config = critic_config
        self.critic_optimizer_config = critic_optimizer_config
        self.agent_optimizer_config = agent_optimizer_config
        self.ssl_model_config = ssl_model_config
        self.ssl_optimizer_config = ssl_optimizer_config
        self.reconstruction_loss = reconstruction_loss
        self.data_constructor = data_constructor
        self.kl_divergence_weight = kl_divergence_weight
        self.self_supervised_learning_loss_weight = self_supervised_learning_loss_weight
        self.loss_combination_method = loss_combination_method
        self.pit_loss_alpha = pit_loss_alpha
        self.warmup_iterations = warmup_iterations
        self.recon_mode = recon_mode
        self.kl_on_agent = kl_on_agent
        self.kl_on_group = kl_on_group

        super().__init__(
            device_ids=device_ids,
            world_size=len(device_ids),
            init_method=init_method,
        )

    def _get_worker_class(self):
        from marlite.trainer.trainer_worker.vaegc_mappo_worker import (
            VAEGroupConsensusMAPPOWorker,
        )
        return VAEGroupConsensusMAPPOWorker

    def _create_worker_kwargs(self) -> Dict[str, Any]:
        kwargs = {}
        kwargs["agent_group_config"] = self.agent_group_config
        kwargs["critic_config"] = self.critic_config
        kwargs["critic_optimizer_config"] = self.critic_optimizer_config
        kwargs["agent_optimizer_config"] = self.agent_optimizer_config
        kwargs["gamma"] = self.gamma
        kwargs["max_grad_norm"] = self.max_grad_norm
        kwargs["clip_epsilon"] = self.clip_epsilon
        kwargs["gae_lambda"] = self.gae_lambda
        kwargs["entropy_coef"] = self.entropy_coef
        kwargs["vf_coef"] = self.vf_coef
        kwargs["ssl_model_config"] = self.ssl_model_config
        kwargs["ssl_optimizer_config"] = self.ssl_optimizer_config
        kwargs["reconstruction_loss"] = self.reconstruction_loss
        kwargs["data_constructor"] = self.data_constructor
        kwargs["kl_divergence_weight"] = self.kl_divergence_weight
        kwargs["self_supervised_learning_loss_weight"] = self.self_supervised_learning_loss_weight
        kwargs["loss_combination_method"] = self.loss_combination_method
        kwargs["pit_loss_alpha"] = self.pit_loss_alpha
        kwargs["warmup_iterations"] = self.warmup_iterations
        kwargs["recon_mode"] = self.recon_mode
        kwargs["kl_on_agent"] = self.kl_on_agent
        kwargs["kl_on_group"] = self.kl_on_group
        return kwargs

    def train_step(self, batch: Dict[str, Any]) -> tuple:
        batch_slices = _slice_batch(batch, self.world_size)
        for i in range(self.world_size):
            self.cmd_queues[i].put("TRAIN_STEP")
            self.data_queues[i].put(batch_slices[i])

        combined_losses = []
        critic_losses = []
        vae_losses = []
        for _ in range(self.world_size):
            result = self.loss_queue.get()
            if isinstance(result, tuple):
                combined, critic, vae = result
                combined_losses.append(combined)
                critic_losses.append(critic)
                vae_losses.append(vae)
            else:
                combined_losses.append(result)
                critic_losses.append(result)
                vae_losses.append(0.0)

        return (
            sum(combined_losses) / len(combined_losses),
            sum(critic_losses) / len(critic_losses),
            sum(vae_losses) / len(vae_losses),
        )
