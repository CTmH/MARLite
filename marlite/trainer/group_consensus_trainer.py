import torch
import numpy as np
from tqdm import tqdm

from marlite.trainer.trainer import Trainer
from marlite.trainer.trainer_worker_group.group_consensus_worker_group import (
    GroupConsensusWorkerGroup,
)
from marlite.util.trajectory_dataset import TrajectoryDataLoader


class GroupConsensusTrainer(Trainer):
    def __init__(
        self,
        kl_divergence_weight: float = 0.005,
        warmup_epochs: int = 0,
        **kwargs,
    ):
        self.kl_divergence_weight = kl_divergence_weight
        self.warmup_epochs = warmup_epochs
        super().__init__(**kwargs)

    def _create_worker_group(self):
        if not self.use_multi_gpu:
            return None

        return GroupConsensusWorkerGroup(
            device_ids=list(range(len(self.device_list))),
            agent_group_config=self.agent_group_config,
            critic_config=self.critic_config,
            critic_optimizer_config=self.critic_optimizer_config,
            agent_optimizer_config=self.agent_optimizer_config,
            gamma=self.gamma,
            kl_divergence_weight=self.kl_divergence_weight,
            warmup_epochs=self.warmup_epochs,
        )

    def learn(self, sample_size, batch_size: int, times: int = 1):
        if not self.use_multi_gpu:
            return self._learn_single_gpu(sample_size, batch_size, times)
        return self._learn_multi_gpu(sample_size, batch_size, times)

    def _learn_single_gpu(self, sample_size, batch_size: int, times: int = 1):
        total_loss = 0.0
        total_batches = 0
        total_td = 0.0
        total_kl = 0.0

        self.eval_agent_group.to(self.train_device)
        self.eval_critic.to(self.train_device)
        self.target_agent_group.to(self.train_device)
        self.target_critic.to(self.train_device)

        is_warmup = self.current_epoch < self.warmup_epochs

        for t in range(times):
            with tqdm(
                total=sample_size, desc=f"Times {t + 1}/{times}", unit="batch"
            ) as pbar:
                dataset = self.replaybuffer.sample(sample_size)
                dataloader = TrajectoryDataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=self.n_workers,
                )
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
                    next_observations = batch["next_observations"].to(
                        dtype=torch.float32
                    )
                    next_timestep_padding_mask = batch[
                        "next_timestep_padding_mask"
                    ].to(dtype=torch.bool)
                    next_avail_actions = batch["next_avail_actions"]
                    next_alive_mask = batch["next_alive_mask"].to(dtype=torch.bool)
                    terminations = batch["terminations"].to(dtype=torch.bool)
                    bs = states.shape[0]
                    n_agents = rewards.shape[2]

                    next_alive_mask = next_alive_mask.to(self.train_device)
                    alive_mask = alive_mask.to(self.train_device)

                    if isinstance(next_avail_actions, torch.Tensor):
                        use_action_mask = True
                        next_avail_actions = next_avail_actions[:, -1, :, :]
                        next_avail_actions = next_avail_actions.to(
                            dtype=torch.bool, device=self.train_device
                        )
                    else:
                        use_action_mask = False

                    rewards = rewards[:, -1]
                    rewards = rewards.sum(dim=1).to(self.train_device)
                    terminations = terminations[:, -1]
                    terminations = terminations.prod(dim=1).to(self.train_device)

                    timestep_padding_mask = torch.stack(
                        [timestep_padding_mask] * n_agents, dim=1
                    ).to(self.train_device)
                    next_timestep_padding_mask = torch.stack(
                        [next_timestep_padding_mask] * n_agents, dim=1
                    ).to(self.train_device)

                    self.eval_agent_group.reset().train()
                    observations_t = torch.transpose(observations, 1, 2).to(
                        self.train_device
                    )
                    states_t = states.to(self.train_device)

                    states_np = states_t[:, -1].detach().cpu().numpy()

                    ret = self.eval_agent_group(
                        observations_t,
                        states_np,
                        timestep_padding_mask,
                        alive_mask[:, -1, :],
                    )
                    q_val = ret["q_val"]
                    group_mu = ret["group_mu"]
                    group_log_var = ret["group_log_var"]
                    agent_mu = ret["agent_mu"]
                    agent_log_var = ret["agent_log_var"]
                    group_indices = ret["group_indices"]

                    actions_last = actions[:, -1].to(
                        device=self.train_device, dtype=torch.int64
                    )
                    q_val = torch.gather(
                        q_val, dim=-1, index=actions_last.unsqueeze(-1)
                    ).squeeze(-1)

                    self.eval_critic.train()
                    ret_critic = self.eval_critic(
                        q_val,
                        states_t,
                        alive_mask,
                        timestep_padding_mask[:, 0, :],
                        group_mu=group_mu,
                        group_log_var=group_log_var,
                        group_indices=group_indices,
                    )
                    q_tot = ret_critic["q_tot"]

                    with torch.no_grad():
                        self.target_agent_group.reset().eval()
                        next_observations_t = torch.transpose(
                            next_observations, 1, 2
                        ).to(self.train_device)
                        next_states_t = next_states.to(self.train_device)
                        next_states_np = (
                            next_states_t[:, -1].detach().cpu().numpy()
                        )

                        ret_next = self.target_agent_group(
                            next_observations_t,
                            next_states_np,
                            next_timestep_padding_mask,
                            next_alive_mask[:, -1, :],
                        )
                        q_val_next = ret_next["q_val"]
                        group_mu_next = ret_next.get("group_mu")
                        group_log_var_next = ret_next.get("group_log_var")
                        group_indices_next = ret_next.get("group_indices")

                        if use_action_mask:
                            q_val_next = torch.masked_fill(
                                q_val_next, ~next_avail_actions, -torch.inf
                            )
                        q_val_next = q_val_next.max(dim=-1).values

                        self.target_critic.eval()
                        ret_next_critic = self.target_critic(
                            q_val_next,
                            next_states_t,
                            next_alive_mask,
                            next_timestep_padding_mask[:, 0, :],
                            group_mu=group_mu_next,
                            group_log_var=group_log_var_next,
                            group_indices=group_indices_next,
                        )
                        q_tot_next = ret_next_critic["q_tot"]

                    y_tot = rewards + (1 - terminations) * self.gamma * q_tot_next
                    td_error = torch.nn.functional.mse_loss(q_tot, y_tot.detach())

                    # KL divergence: KL(N(μ,σ²) || N(0,1))
                    if is_warmup:
                        kl_divergence = torch.tensor(0.0, device=self.train_device)
                    else:
                        kl_divergence = -0.5 * torch.sum(
                            1 + agent_log_var - agent_mu.pow(2) - torch.exp(agent_log_var),
                            dim=-1,
                        )
                        kl_divergence = torch.mean(kl_divergence)

                    critic_loss = (
                        td_error + self.kl_divergence_weight * kl_divergence
                    )

                    self.agent_optimizer.zero_grad()
                    self.eval_critic.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.eval_critic.parameters(), max_norm=5.0
                    )
                    torch.nn.utils.clip_grad_norm_(
                        self.eval_agent_group.parameters(), max_norm=5.0
                    )
                    self.critic_optimizer.step()
                    self.agent_optimizer.step()

                    total_loss += critic_loss.detach().cpu().item()
                    total_td += td_error.detach().cpu().item()
                    total_kl += kl_divergence.detach().cpu().item()
                    total_batches += 1
                    pbar.update(bs)

        self.eval_agent_group.to("cpu")
        self.eval_critic.to("cpu")
        self.target_agent_group.to("cpu")
        self.target_critic.to("cpu")
        torch.cuda.empty_cache()

        avg_loss = total_loss / total_batches
        avg_td = total_td / total_batches
        avg_kl = total_kl / total_batches

        return avg_loss

    def _learn_multi_gpu(self, sample_size, batch_size: int, times: int = 1):
        self.worker_group.move_models_to_gpu()

        total_loss = 0.0
        total_batches = 0

        for t in range(times):
            with tqdm(
                total=sample_size, desc=f"Times {t + 1}/{times}", unit="batch"
            ) as pbar:
                dataset = self.replaybuffer.sample(sample_size)
                dataloader = TrajectoryDataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=self.n_workers,
                )
                for batch in dataloader:
                    batch["epoch"] = self.current_epoch
                    loss = self.worker_group.train_step(batch)
                    total_loss += loss
                    total_batches += 1
                    bs = batch["states"].shape[0]
                    pbar.update(bs)

        self.worker_group.move_models_to_cpu()
        torch.cuda.empty_cache()

        return total_loss / total_batches
