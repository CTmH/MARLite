import numpy as np
import torch
from tqdm import tqdm

from marlite.trainer.offpolicy_trainer import OffPolicyTrainer
from marlite.trainer.trainer_worker_group import QMIXWorkerGroup
from marlite.util.trajectory_dataset import TrajectoryDataLoader


class QMIXTrainer(OffPolicyTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _create_worker_group(self):
        """Create QMIXWorkerGroup for multi-GPU training."""
        if not self.use_multi_gpu:
            return None

        return QMIXWorkerGroup(
            device_ids=self._get_device_ids(),
            agent_group_config=self.agent_group_config,
            critic_config=self.critic_config,
            critic_optimizer_config=self.critic_optimizer_config,
            agent_optimizer_config=self.agent_optimizer_config,
            gamma=self.gamma,
            max_grad_norm=self.max_grad_norm,
        )

    def learn(self, sample_size, batch_size: int, times: int = 1):
        if not self.use_multi_gpu:
            return self._learn_single_gpu(sample_size, batch_size, times)
        return self._learn_multi_gpu(sample_size, batch_size, times)

    def _learn_single_gpu(self, sample_size, batch_size: int, times: int = 1):
        """Single GPU learning."""
        total_loss = 0.0
        total_batches = 0

        self.eval_agent_group.to(self.train_device)
        self.eval_critic.to(self.train_device)
        self.target_agent_group.to(self.train_device)
        self.target_critic.to(self.train_device)

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
                    next_timestep_padding_mask = batch["next_timestep_padding_mask"].to(
                        dtype=torch.bool
                    )
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

                    r_last = self._aggregate_rewards(rewards[:, -1]).to(self.train_device)
                    termination_last = terminations[:, -1].prod(dim=1).to(self.train_device)

                    timestep_padding_mask = torch.stack(
                        [timestep_padding_mask] * n_agents, dim=1
                    ).to(self.train_device)
                    next_timestep_padding_mask = torch.stack(
                        [next_timestep_padding_mask] * n_agents, dim=1
                    ).to(self.train_device)

                    self.eval_agent_group.train()
                    observations = torch.transpose(observations, 1, 2).to(
                        self.train_device
                    )
                    ret = self.eval_agent_group(
                        observations, timestep_padding_mask, alive_mask[:, -1, :]
                    )
                    q_val = ret["q_val"]
                    actions = actions[:, -1].to(
                        device=self.train_device, dtype=torch.int64
                    )
                    q_val = torch.gather(q_val, dim=-1, index=actions.unsqueeze(-1))
                    q_val = q_val.squeeze(-1)
                    states = states.to(self.train_device)
                    self.eval_critic.train()
                    ret = self.eval_critic(
                        q_val, states, alive_mask, timestep_padding_mask[:, 0, :]
                    )
                    q_tot = ret["q_tot"]

                    with torch.no_grad():
                        # Double Q: eval agent group selects best actions
                        self.eval_agent_group.eval()
                        next_observations = torch.transpose(next_observations, 1, 2).to(
                            self.train_device
                        )
                        ret_next_eval = self.eval_agent_group(
                            next_observations,
                            next_timestep_padding_mask,
                            next_alive_mask[:, -1, :],
                        )
                        q_val_next_eval = ret_next_eval["q_val"]
                        if use_action_mask:
                            q_val_next_eval = torch.masked_fill(
                                q_val_next_eval, ~next_avail_actions, -torch.inf
                            )
                        best_actions = q_val_next_eval.argmax(dim=-1)

                        # Double Q: target agent group evaluates chosen actions
                        self.target_agent_group.eval()
                        ret_next_target = self.target_agent_group(
                            next_observations,
                            next_timestep_padding_mask,
                            next_alive_mask[:, -1, :],
                        )
                        q_val_next = ret_next_target["q_val"].gather(
                            dim=-1, index=best_actions.unsqueeze(-1)
                        ).squeeze(-1)
                        next_states = next_states.to(self.train_device)
                        self.target_critic.eval()
                        ret_next = self.target_critic(
                            q_val_next,
                            next_states,
                            next_alive_mask,
                            next_timestep_padding_mask[:, 0, :],
                        )
                        q_tot_next = ret_next["q_tot"]

                    y_tot = r_last + (1 - termination_last) * self.gamma * q_tot_next
                    critic_loss = torch.nn.functional.mse_loss(q_tot, y_tot.detach())

                    self.agent_optimizer.zero_grad()
                    self.eval_critic.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.eval_critic.parameters(), max_norm=self.max_grad_norm
                    )
                    torch.nn.utils.clip_grad_norm_(
                        self.eval_agent_group.parameters(), max_norm=self.max_grad_norm
                    )
                    self.critic_optimizer.step()
                    self.agent_optimizer.step()

                    total_loss += critic_loss.detach().cpu().item()
                    total_batches += 1

                    pbar.update(bs)

        self.eval_agent_group.to("cpu")
        self.eval_critic.to("cpu")
        self.target_agent_group.to("cpu")
        self.target_critic.to("cpu")

        torch.cuda.empty_cache()

        return total_loss / total_batches

    def _learn_multi_gpu(self, sample_size, batch_size: int, times: int = 1):
        """Multi-GPU learning via worker processes."""
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
