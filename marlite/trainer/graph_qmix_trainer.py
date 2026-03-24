import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from marlite.trainer.trainer import Trainer
from marlite.util.trajectory_dataset import TrajectoryDataLoader
from marlite.util.distributed_utils import get_local_device_id, average_loss


class GraphQMIXTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Agent group

    def learn(self, sample_size, batch_size: int, times: int = 1):
        total_loss = 0.0
        total_batches = 0

        # Check if DDP is enabled
        if self.use_ddp:
            # Multi-GPU DDP training
            device_id = get_local_device_id(self.train_device)

            # Move models to device and wrap with DDP
            self.eval_agent_group.wrap_data_parallel(device_id)
            self.eval_critic = DDP(
                self.eval_critic.to(self.train_device), device_ids=[device_id]
            )
            self.target_agent_group.wrap_data_parallel(device_id)
            self.target_critic = DDP(
                self.target_critic.to(self.train_device), device_ids=[device_id]
            )
        else:
            # Single device training
            self.eval_agent_group.to(self.train_device)
            self.eval_critic.to(self.train_device)
            self.target_agent_group.to(self.train_device)
            self.target_critic.to(self.train_device)

        for t in range(times):
            with tqdm(
                total=sample_size, desc=f"Times {t + 1}/{times}", unit="batch"
            ) as pbar:
                # Implement the learning logic for QMix
                # Get a batch of data from the replay buffer
                dataset = self.replaybuffer.sample(sample_size)
                dataloader = TrajectoryDataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=self.n_workers,
                )
                for batch in dataloader:
                    # Extract batch data - now all tensors except next_avail_actions which could be numpy or tensor
                    alive_mask = batch["alive_mask"].to(dtype=torch.bool)  # (B, T, N)
                    observations = batch["observations"].to(
                        dtype=torch.float32
                    )  # (B, T, N, F)
                    timestep_padding_mask = batch["timestep_padding_mask"].to(
                        dtype=torch.bool
                    )  # (B, T)
                    states = batch["states"].to(dtype=torch.float32)  # (B, T, F)
                    edge_indices = batch["edge_indices"]  # (B, T, 2, N)
                    actions = batch["actions"].to(dtype=torch.int)  # (B, T, N)
                    rewards = batch["rewards"].to(dtype=torch.float32)  # (B, T, N)
                    next_states = batch["next_states"].to(
                        dtype=torch.float32
                    )  # (B, T, F)
                    next_edge_indices = batch["next_edge_indices"]  # (B, T, 2, N)
                    next_observations = batch["next_observations"].to(
                        dtype=torch.float32
                    )  # (B, T, N, F)
                    next_timestep_padding_mask = batch["next_timestep_padding_mask"].to(
                        dtype=torch.bool
                    )  # (B, T)
                    next_avail_actions = batch[
                        "next_avail_actions"
                    ]  # Could be numpy array or tensor
                    next_alive_mask = batch["next_alive_mask"].to(dtype=torch.bool)
                    terminations = batch["terminations"].to(
                        dtype=torch.bool
                    )  # (B, T, N)
                    # truncations = batch['truncations'].to(dtype=torch.bool)  # (B, T, N)
                    bs = states.shape[0]  # Actual batch size
                    n_agents = rewards.shape[
                        2
                    ]  # Changed from shape[1] to shape[2] since (B, T, N)

                    # Create alive_mask_next from terminations and truncations
                    next_alive_mask = next_alive_mask.to(self.train_device)
                    alive_mask = alive_mask.to(self.train_device)

                    # Action mask: (B, T, N, Actions) -> (B, N, Actions)
                    if isinstance(next_avail_actions, torch.Tensor):
                        use_action_mask = True
                        next_avail_actions = next_avail_actions[
                            :, -1, :, :
                        ]  # (B, T, N, Actions) -> (B, N, Actions)
                        next_avail_actions = next_avail_actions.to(
                            dtype=torch.bool, device=self.train_device
                        )
                    else:
                        use_action_mask = False

                    rewards = rewards[:, -1]  # (B, T, N) -> (B, N)
                    rewards = rewards.sum(dim=1).to(
                        self.train_device
                    )  # (B, N) -> (B) Sum over all agents rewards
                    terminations = terminations[:, -1]  # (B, T, N) -> (B, N)
                    terminations = terminations.prod(dim=1).to(
                        self.train_device
                    )  # (B, N) -> (B) if all agents are terminated then game over

                    # timestep_padding_mask = torch.tensor(timestep_padding_mask, dtype=torch.bool) # (B, T) # REMOVED: already converted above
                    timestep_padding_mask = torch.stack(
                        [timestep_padding_mask] * n_agents, dim=1
                    ).to(self.train_device)  # (B, N, T)
                    # next_timestep_padding_mask = torch.tensor(next_timestep_padding_mask, dtype=torch.bool) # REMOVED: already converted above
                    next_timestep_padding_mask = torch.stack(
                        [next_timestep_padding_mask] * n_agents, dim=1
                    ).to(self.train_device)

                    # Compute the Q-tot
                    last_edge_indices = [
                        edge_indices[i][-1] for i in range(bs)
                    ]  # (B, T, 2, N) -> (B, 2, N) Take only the last edge indices
                    last_next_edge_indices = [
                        next_edge_indices[i][-1] for i in range(bs)
                    ]  # (B, T, 2, N) -> (B, 2, N)
                    # observations = torch.tensor(observations, dtype=torch.float, device=self.train_device) # REMOVED: already converted above
                    self.eval_agent_group.reset().train()  # Reset Graph Builder intervals
                    observations = torch.transpose(observations, 1, 2).to(
                        self.train_device
                    )  # obs.shape (B, T, N, F) -> (B, N, T, F)
                    states = states.to(self.train_device)
                    ret = self.eval_agent_group.forward(
                        observations,
                        states,
                        timestep_padding_mask,
                        alive_mask[:, -1, :],
                        last_edge_indices,
                    )
                    q_val = ret["q_val"]
                    actions = actions[:, -1].to(
                        device=self.train_device, dtype=torch.int64
                    )  # (B, T, N) -> (B, N) # REMOVED torch.Tensor wrapper
                    q_val = torch.gather(q_val, dim=-1, index=actions.unsqueeze(-1))
                    q_val = q_val.squeeze(-1)  # (B, N, 1) -> (B, N)
                    self.eval_critic.train()
                    ret = self.eval_critic(
                        q_val, states, alive_mask, timestep_padding_mask[:, 0, :]
                    )
                    q_tot = ret["q_tot"]

                    # Double Q-learning, we use eval agent group to choose actions,and use target critic to compute q_target
                    with torch.no_grad():
                        # next_observations = torch.tensor(next_observations, dtype=torch.float, device=self.train_device) # REMOVED: already converted above
                        self.target_agent_group.reset().eval()  # Reset Graph Builder intervals
                        next_observations = torch.transpose(next_observations, 1, 2).to(
                            self.train_device
                        )  # obs.shape (B, T, N, F) -> (B, N, T, F)
                        next_states = next_states.to(self.train_device)
                        ret_next = self.target_agent_group.forward(
                            next_observations,
                            next_states,
                            next_timestep_padding_mask,
                            next_alive_mask[:, -1, :],
                            last_next_edge_indices,
                        )
                        q_val_next = ret_next["q_val"]
                        if use_action_mask:
                            q_val_next = torch.masked_fill(
                                q_val_next, ~next_avail_actions, -torch.inf
                            )
                        q_val_next = q_val_next.max(dim=-1).values
                        self.target_critic.eval()
                        ret_next = self.target_critic(
                            q_val_next,
                            next_states,
                            next_alive_mask,
                            next_timestep_padding_mask[:, 0, :],
                        )
                        q_tot_next = ret_next["q_tot"]

                    # Compute the TD target
                    y_tot = rewards + (1 - terminations) * self.gamma * q_tot_next

                    # Compute the critic loss
                    critic_loss = torch.nn.functional.mse_loss(q_tot, y_tot.detach())
                    if self.use_ddp:
                        # Average loss across all processes
                        critic_loss = average_loss(critic_loss, len(self.device_list))

                    # Optimize the critic network
                    self.eval_agent_group.zero_grad()
                    self.eval_critic.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.eval_critic.parameters(), max_norm=5.0
                    )
                    self.optimizer.step()
                    self.eval_agent_group.step()

                    total_loss += critic_loss.detach().cpu().item()
                    total_batches += 1

                    pbar.update(bs)

        if self.use_ddp:
            self.eval_agent_group.unwrap_data_parallel()
            self.eval_critic = self.eval_critic.module.cpu()
            self.target_agent_group.unwrap_data_parallel()
            self.target_critic = self.target_critic.module.cpu()
        else:
            self.eval_agent_group.to("cpu")
            self.eval_critic.to("cpu")
            self.target_agent_group.to("cpu")
            self.target_critic.to("cpu")

        torch.cuda.empty_cache()

        return total_loss / total_batches
