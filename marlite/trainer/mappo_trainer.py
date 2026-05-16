import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from absl import logging
from typing import Union

from marlite.trainer.onpolicy_trainer import OnPolicyTrainer
from marlite.util.trajectory_dataset import TrajectoryDataLoader


class MAPPOTrainer(OnPolicyTrainer):
    def __init__(
        self,
        clip_epsilon: float = 0.2,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 5.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.clip_epsilon = clip_epsilon
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

    def _create_worker_group(self):
        return None

    def learn(self, sample_size, batch_size: int, ppo_epochs: int = 4):
        if not self.use_multi_gpu:
            return self._learn_single_gpu(sample_size, batch_size, ppo_epochs)
        return self._learn_multi_gpu(sample_size, batch_size, ppo_epochs)

    def _learn_single_gpu(self, sample_size, batch_size: int, ppo_epochs: int = 4):
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_batches = 0

        self.eval_agent_group.to(self.train_device)
        self.eval_critic.to(self.train_device)

        dataset = self.replaybuffer.sample(sample_size)
        dataloader = TrajectoryDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.n_workers,
        )

        for epoch in range(ppo_epochs):
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
                next_timestep_padding_mask = batch["next_timestep_padding_mask"].to(
                    dtype=torch.bool
                )
                next_alive_mask = batch["next_alive_mask"].to(dtype=torch.bool)
                all_log_probs = batch["all_log_probs"].to(dtype=torch.float32)
                terminations = batch["terminations"].to(dtype=torch.bool)

                bs = states.shape[0]
                n_agents = rewards.shape[2]
                t_steps = rewards.shape[1]

                device = self.train_device
                alive_mask = alive_mask.to(device)
                next_alive_mask = next_alive_mask.to(device)
                states_dev = states.to(device)
                next_states_dev = next_states.to(device)

                rewards_sum = rewards.sum(dim=2).to(device)
                terminations_any = terminations.any(dim=2).to(
                    dtype=torch.float32, device=device
                )

                timestep_padding_mask_expanded = torch.stack(
                    [timestep_padding_mask] * n_agents, dim=1
                ).to(device)

                self.eval_critic.train()
                ret_critic = self.eval_critic(
                    states_dev, alive_mask, timestep_padding_mask
                )
                v_all = ret_critic["v"].squeeze(-1)

                with torch.no_grad():
                    next_states_dev = next_states.to(device)
                    ret_critic_next = self.eval_critic(
                        next_states_dev[:, -1:, ...],
                        next_alive_mask[:, -1:, :],
                        next_timestep_padding_mask[:, -1:],
                    )
                    v_next_bootstrap = ret_critic_next["v"].squeeze(-1)[:, 0]

                timestep_valid = (~timestep_padding_mask).to(
                    dtype=torch.float32, device=device
                )

                v_next_padded = torch.cat(
                    [v_all[:, 1:], v_next_bootstrap.unsqueeze(1)], dim=1
                )
                delta = (
                    rewards_sum
                    + self.gamma * v_next_padded * (1.0 - terminations_any)
                    - v_all
                )

                advantages = torch.zeros_like(rewards_sum)
                gae = torch.zeros(bs, device=device)
                for t in reversed(range(t_steps)):
                    gae = delta[:, t] + self.gamma * self.gae_lambda * (
                        1.0 - terminations_any[:, t]
                    ) * gae
                    gae = gae * timestep_valid[:, t]
                    advantages[:, t] = gae
                returns = advantages + v_all

                valid_count = timestep_valid.sum()
                if valid_count > 0:
                    adv_mean = (advantages * timestep_valid).sum() / valid_count
                    adv_var = (
                        ((advantages - adv_mean) ** 2) * timestep_valid
                    ).sum() / valid_count
                    advantages = (advantages - adv_mean) / (adv_var.sqrt() + 1e-8)

                observations_transposed = torch.transpose(observations, 1, 2).to(
                    device
                )
                self.eval_agent_group.train()
                ret_agent = self.eval_agent_group(
                    observations_transposed,
                    timestep_padding_mask_expanded,
                    alive_mask[:, -1, :],
                )
                action_logits = ret_agent["action_logits"]

                actions_last = actions[:, -1].to(dtype=torch.int64, device=device)
                log_probs_old = all_log_probs[:, -1, :].to(device)

                dist = Categorical(logits=action_logits)
                new_log_probs = dist.log_prob(actions_last)
                entropy = dist.entropy()

                alive_last_flag = alive_mask[:, -1, :].to(
                    dtype=torch.float32, device=device
                )
                alive_last_count = alive_last_flag.sum()

                ratio = torch.exp(new_log_probs - log_probs_old)
                advantages_last = advantages[:, -1].unsqueeze(-1).expand(-1, n_agents)
                surr1 = ratio * advantages_last
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
                    )
                    * advantages_last
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

                critic_loss_mse = F.mse_loss(v_all, returns, reduction="none")
                critic_loss = (critic_loss_mse * timestep_valid).sum() / max(
                    valid_count, torch.tensor(1.0, device=device)
                )

                self.agent_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(
                    self.eval_agent_group.parameters(), max_norm=self.max_grad_norm
                )

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.eval_critic.parameters(), max_norm=self.max_grad_norm
                )

                self.agent_optimizer.step()
                self.critic_optimizer.step()

                total_actor_loss += actor_loss.detach().cpu().item()
                total_critic_loss += critic_loss.detach().cpu().item()
                total_batches += 1

        self.eval_agent_group.to("cpu")
        self.eval_critic.to("cpu")
        torch.cuda.empty_cache()

        avg_actor = total_actor_loss / max(total_batches, 1)
        avg_critic = total_critic_loss / max(total_batches, 1)
        return avg_actor + avg_critic * self.vf_coef

    def _learn_multi_gpu(self, sample_size, batch_size: int, ppo_epochs: int = 4):
        raise NotImplementedError("Multi-GPU not implemented for MAPPO")

    def train(
        self,
        iterations,
        target_first_metric,
        batch_size=64,
        ppo_epochs=4,
    ):
        self.eval_episodes_to_replay_ratio = 1.0

        self.evaluate()

        for iteration in range(iterations):
            self.current_epoch = iteration

            sample_size = len(self.replaybuffer.buffer)
            if sample_size > 0:
                agent_group_lr = self.agent_optimizer.param_groups[0]["lr"]
                critic_lr = self.critic_optimizer.param_groups[0]["lr"]
                logging.info(
                    f"Iteration {iteration}: Batch size: {batch_size}, "
                    f"Critic lr: {critic_lr:.8f}, Agent lr: {agent_group_lr:.8f}"
                )
                loss = self.learn(
                    sample_size=sample_size,
                    batch_size=batch_size,
                    ppo_epochs=ppo_epochs,
                )
                logging.info(f"Iteration {iteration}: Loss {loss:.4f}")

            self.replaybuffer = self.replaybuffer_config.create_replaybuffer()

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
                self.save_current_model(checkpoint="best")

            if first_metric >= target_first_metric:
                break

        return self.best_metrics
