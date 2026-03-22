import numpy as np
import torch
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributions import Normal, kl_divergence

from marlite.trainer.trainer import Trainer
from marlite.util.trajectory_dataset import TrajectoryDataLoader
from marlite.util.loss_func import PITLoss
from marlite.util.distributed_utils import get_local_device_id, average_loss


class MsgAggrQMIXTrainer(Trainer):
    def __init__(self, **kwargs):
        margin = kwargs.pop("triplet_loss_margin", 1.0)
        pit_loss_alpha = kwargs.pop("pit_loss_alpha", 0.9)
        cosine_margin = kwargs.pop("cosine_margin", 0.5)
        self.warmup_epochs = kwargs.pop("warmup_epochs", 0)
        # Add loss function selection parameters
        loss_type = kwargs.pop("loss_type", "weighted_sum")  # 'pit' or 'weighted_sum'
        self.msg_aggr_weight = kwargs.pop("msg_aggr_weight", 1.0)
        super().__init__(**kwargs)
        self.triplet_loss = torch.nn.TripletMarginLoss(margin=margin)
        self.pit_loss = PITLoss(num_tasks=2, alpha=pit_loss_alpha)
        self.cosine_embedding_loss = torch.nn.CosineEmbeddingLoss(
            margin=cosine_margin, reduction="mean"
        )

        # Determine which loss function to use based on configuration
        if loss_type == "pit":
            self.compute_critic_loss = self._compute_pit_loss
        elif loss_type == "weighted_sum":
            self.compute_critic_loss = self._compute_weighted_sum_loss
        else:
            raise ValueError(
                f"Unknown loss_type: {loss_type}. Supported values are 'pit' and 'weighted_sum'."
            )

    def _compute_pit_loss(self, td_error, msg_aggr_loss):
        """Compute loss using PIT loss function."""
        self.pit_loss.to(self.train_device)
        return self.pit_loss(torch.stack([td_error, msg_aggr_loss]))

    def _compute_weighted_sum_loss(self, td_error, msg_aggr_loss):
        """Compute loss using weighted sum of individual losses."""
        return td_error + self.msg_aggr_weight * msg_aggr_loss

    def learn(self, sample_size, batch_size: int, times: int = 1):
        total_loss = 0.0
        total_batches = 0

        # Move models to the appropriate device before wrapping with DataParallel
        self.eval_agent_group.to(self.train_device)
        self.eval_critic.to(self.train_device)
        self.target_agent_group.to(self.train_device)
        self.target_critic.to(self.train_device)

        if self.use_ddp:
            self.eval_agent_group.wrap_data_parallel()
            device_id = get_local_device_id(self.train_device)
            self.eval_critic = DDP(self.eval_critic, device_ids=[device_id])
            self.target_agent_group.wrap_data_parallel()
            self.target_critic = DDP(self.target_critic, device_ids=[device_id])

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
                    # Extract batch data - now all tensors except next_avail_actions which might be numpy array or tensor
                    alive_mask = batch["alive_mask"].to(dtype=torch.bool)
                    observations = batch["observations"].to(dtype=torch.float32)
                    obs_padding_mask = batch["obs_padding_mask"].to(dtype=torch.bool)
                    states = batch["states"].to(dtype=torch.float32)
                    actions = batch["actions"].to(dtype=torch.int)
                    rewards = batch["rewards"].to(dtype=torch.float32)
                    next_states = batch["next_states"].to(dtype=torch.float32)
                    next_observations = batch["next_observations"].to(
                        dtype=torch.float32
                    )
                    next_obs_padding_mask = batch["next_obs_padding_mask"].to(
                        dtype=torch.bool
                    )
                    next_avail_actions = batch[
                        "next_avail_actions"
                    ]  # Numpy array or Tensor
                    terminations = batch["terminations"].to(dtype=torch.bool)
                    truncations = batch["truncations"].to(dtype=torch.bool)
                    bs = states.shape[0]  # Actual batch size
                    n_agents = rewards.shape[2]

                    # Create alive_mask_next from terminations and truncations
                    # alive_mask = torch.tensor(alive_mask).to(dtype=torch.bool) # (B, T, N) # REMOVED: already converted above
                    terminations = terminations[
                        :, -1
                    ]  # (B, T, N) -> (B, N) # REMOVED torch.tensor conversion
                    truncations = truncations[
                        :, -1
                    ]  # (B, T, N) -> (B, N) # REMOVED torch.tensor conversion
                    next_alive_mask = ~(terminations | truncations)
                    next_alive_mask = next_alive_mask.unsqueeze(dim=1)
                    next_alive_mask = torch.cat(
                        [alive_mask[:, 1:, :], next_alive_mask], dim=1
                    )
                    next_alive_mask = next_alive_mask.to(self.train_device)
                    alive_mask = alive_mask.to(self.train_device)
                    # Action mask: (B, T, N, Actions) -> (B, N, Actions) - next_avail_actions has same dimension structure
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
                    terminations = terminations.prod(dim=1).to(
                        self.train_device
                    )  # (B, N) -> (B) if all agents are terminated then game over

                    # obs_padding_mask = torch.tensor(obs_padding_mask, dtype=torch.bool) # (B, T) # REMOVED: already converted above
                    obs_padding_mask = torch.stack(
                        [obs_padding_mask] * n_agents, dim=1
                    ).to(self.train_device)  # (B, N, T)
                    # next_obs_padding_mask = torch.tensor(next_obs_padding_mask, dtype=torch.bool) # REMOVED: already converted above
                    next_obs_padding_mask = torch.stack(
                        [next_obs_padding_mask] * n_agents, dim=1
                    ).to(self.train_device)

                    # Compute the Q-tot
                    self.eval_agent_group.train()
                    observations = torch.transpose(observations, 1, 2).to(
                        self.train_device
                    )  # obs.shape (B, T, N, F) -> (B, N, T, F)
                    ret = self.eval_agent_group.forward(
                        observations, obs_padding_mask, alive_mask[:, -1, :]
                    )
                    q_val = ret["q_val"]
                    aggregated_msg = ret["aggregated_msg"]
                    actions = actions[:, -1].to(
                        device=self.train_device, dtype=torch.int64
                    )  # (B, T, N) -> (B, N) # REMOVED torch.Tensor wrapper
                    q_val = torch.gather(q_val, dim=-1, index=actions.unsqueeze(-1))
                    q_val = q_val.squeeze(-1)  # (B, N, 1) -> (B, N)
                    states = states.to(self.train_device)
                    self.eval_critic.train()
                    ret = self.eval_critic(
                        q_val, states, alive_mask, obs_padding_mask[:, 0, :]
                    )
                    q_tot = ret["q_tot"]
                    # Use target model for stablity
                    with torch.no_grad():
                        self.target_critic.eval()
                        ret = self.target_critic(
                            q_val, states, alive_mask, obs_padding_mask[:, 0, :]
                        )
                        state_features = ret["state_features"]

                    # Compute TD targets
                    with torch.no_grad():
                        self.target_agent_group.eval()
                        next_observations = torch.transpose(next_observations, 1, 2).to(
                            self.train_device
                        )  # obs.shape (B, T, N, F) -> (B, N, T, F)
                        ret_next = self.eval_agent_group.forward(
                            next_observations,
                            next_obs_padding_mask,
                            next_alive_mask[:, -1, :],
                        )
                        q_val_next = ret_next["q_val"]
                        if use_action_mask:
                            q_val_next = torch.masked_fill(
                                q_val_next, ~next_avail_actions, -torch.inf
                            )
                        q_val_next = q_val_next.max(dim=-1).values
                        next_states = next_states.to(self.train_device)
                        self.target_critic.eval()
                        ret_next = self.target_critic(
                            q_val_next,
                            next_states,
                            next_alive_mask,
                            next_obs_padding_mask[:, 0, :],
                        )
                        q_tot_next = ret_next["q_tot"]
                        # state_features_next = ret_next['state_features']

                    # Compute the TD target
                    y_tot = rewards + (1 - terminations) * self.gamma * q_tot_next

                    # TD error
                    td_error = torch.nn.functional.mse_loss(q_tot, y_tot.detach())

                    # Only compute message aggregation losses after warmup period
                    if self.current_epoch >= self.warmup_epochs:
                        # Message aggregation loss
                        msg_aggr_loss = torch.functional.F.smooth_l1_loss(
                            aggregated_msg, state_features.detach()
                        )

                        # Use the predetermined loss function
                        critic_loss = self.compute_critic_loss(td_error, msg_aggr_loss)
                    else:
                        # Before warmup period: only use TD error
                        critic_loss = td_error

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


class ProbMsgAggrQMIXTrainer(Trainer):
    def __init__(self, **kwargs):
        pit_loss_alpha = kwargs.pop("pit_loss_alpha", 0.9)
        # Add loss function selection parameters
        loss_type = kwargs.pop("loss_type", "weighted_sum")  # 'pit' or 'weighted_sum'
        self.msg_aggr_weight = kwargs.pop("msg_aggr_weight", 1.0)
        self.warmup_epochs = kwargs.pop("warmup_epochs", 0)
        super().__init__(**kwargs)
        self.pit_loss = PITLoss(num_tasks=2, alpha=pit_loss_alpha)

        # Determine which loss function to use based on configuration
        if loss_type == "pit":
            self.compute_critic_loss = self._compute_pit_loss
        elif loss_type == "weighted_sum":
            self.compute_critic_loss = self._compute_weighted_sum_loss
        else:
            raise ValueError(
                f"Unknown loss_type: {loss_type}. Supported values are 'pit' and 'weighted_sum'."
            )

    def _compute_pit_loss(self, td_error, msg_aggr_loss):
        """Compute loss using PIT loss function."""
        self.pit_loss.to(self.train_device)
        return self.pit_loss(torch.stack([td_error, msg_aggr_loss]))

    def _compute_weighted_sum_loss(self, td_error, msg_aggr_loss):
        """Compute loss using weighted sum of individual losses."""
        return td_error + self.msg_aggr_weight * msg_aggr_loss

    def learn(self, sample_size, batch_size: int, times: int = 1):
        total_loss = 0.0
        total_batches = 0

        # Move models to the appropriate device before wrapping with DataParallel
        self.eval_agent_group.to(self.train_device)
        self.eval_critic.to(self.train_device)
        self.target_agent_group.to(self.train_device)
        self.target_critic.to(self.train_device)

        if self.use_ddp:
            self.eval_agent_group.wrap_data_parallel()
            device_id = get_local_device_id(self.train_device)
            self.eval_critic = DDP(self.eval_critic, device_ids=[device_id])
            self.target_agent_group.wrap_data_parallel()
            self.target_critic = DDP(self.target_critic, device_ids=[device_id])

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
                    # Extract batch data
                    alive_mask = batch["alive_mask"].to(dtype=torch.bool)
                    observations = batch["observations"].to(dtype=torch.float32)
                    obs_padding_mask = batch["obs_padding_mask"].to(dtype=torch.bool)
                    states = batch["states"].to(dtype=torch.float32)
                    actions = batch["actions"].to(dtype=torch.int)
                    rewards = batch["rewards"].to(dtype=torch.float32)
                    next_states = batch["next_states"].to(dtype=torch.float32)
                    next_observations = batch["next_observations"].to(
                        dtype=torch.float32
                    )
                    next_obs_padding_mask = batch["next_obs_padding_mask"].to(
                        dtype=torch.bool
                    )
                    next_avail_actions = batch[
                        "next_avail_actions"
                    ]  # Numpy array or Tensor
                    next_alive_mask = batch["next_alive_mask"].to(dtype=torch.bool)
                    terminations = batch["terminations"].to(dtype=torch.bool)
                    truncations = batch["truncations"].to(dtype=torch.bool)
                    bs = states.shape[0]  # Actual batch size
                    n_agents = rewards.shape[2]

                    # Create alive_mask_next from terminations and truncations
                    next_alive_mask = next_alive_mask.to(self.train_device)
                    alive_mask = alive_mask.to(self.train_device)
                    # Action mask: (B, T, N, Actions) -> (B, N, Actions) - next_avail_actions has same dimension structure
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

                    # obs_padding_mask = torch.tensor(obs_padding_mask, dtype=torch.bool) # (B, T) # REMOVED: already converted above
                    obs_padding_mask = torch.stack(
                        [obs_padding_mask] * n_agents, dim=1
                    ).to(self.train_device)  # (B, N, T)
                    # next_obs_padding_mask = torch.tensor(next_obs_padding_mask, dtype=torch.bool) # REMOVED: already converted above
                    next_obs_padding_mask = torch.stack(
                        [next_obs_padding_mask] * n_agents, dim=1
                    ).to(self.train_device)

                    # Compute the Q-tot
                    self.eval_agent_group.train()
                    observations = torch.transpose(observations, 1, 2).to(
                        self.train_device
                    )  # obs.shape (B, T, N, F) -> (B, N, T, F)
                    ret = self.eval_agent_group.forward(
                        observations, obs_padding_mask, alive_mask[:, -1, :]
                    )  # obs.shape (B, N, T, F)
                    q_val = ret["q_val"]
                    ag_mu = ret["mu"]
                    ag_std = ret["std"]
                    actions = actions[:, -1].to(
                        device=self.train_device, dtype=torch.int64
                    )  # (B, T, N) -> (B, N) # REMOVED torch.Tensor wrapper
                    q_val = torch.gather(q_val, dim=-1, index=actions.unsqueeze(-1))
                    q_val = q_val.squeeze(-1)  # (B, N, 1) -> (B, N)
                    states = states.to(self.train_device)
                    self.eval_critic.train()
                    ret = self.eval_critic(
                        q_val, states, alive_mask, obs_padding_mask[:, 0, :]
                    )
                    q_tot = ret["q_tot"]
                    # Use target model for stablity
                    with torch.no_grad():
                        self.target_critic.eval()
                        ret = self.target_critic(
                            q_val, states, alive_mask, obs_padding_mask[:, 0, :]
                        )
                        critic_mu = ret["mu"]
                        critic_std = ret["std"]

                    # state_dict = {'ag_mu': ag_mu, 'ag_std': ag_std, 'critic_mu': critic_mu, 'critic_std': critic_std}
                    # torch.save(state_dict, '/home/ctmh/Source/MARLite/draft/multiple_tensors.pth')

                    # Compute TD targets
                    with torch.no_grad():
                        self.target_agent_group.eval()
                        next_observations = torch.transpose(next_observations, 1, 2).to(
                            self.train_device
                        )  # obs.shape (B, T, N, F) -> (B, N, T, F)
                        ret_next = self.target_agent_group.forward(
                            next_observations,
                            next_obs_padding_mask,
                            next_alive_mask[:, -1, :],
                        )
                        q_val_next = ret_next["q_val"]
                        if use_action_mask:
                            q_val_next = torch.masked_fill(
                                q_val_next, ~next_avail_actions, -torch.inf
                            )
                        q_val_next = q_val_next.max(dim=-1).values
                        next_states = next_states.to(self.train_device)
                        self.target_critic.eval()
                        ret_next = self.target_critic(
                            q_val_next,
                            next_states,
                            next_alive_mask,
                            next_obs_padding_mask[:, 0, :],
                        )
                        q_tot_next = ret_next["q_tot"]

                    # Compute the TD target
                    y_tot = rewards + (1 - terminations) * self.gamma * q_tot_next

                    # TD error
                    td_error = torch.nn.functional.mse_loss(q_tot, y_tot.detach())
                    # Message aggregation loss
                    ag_distribution = Normal(ag_mu, ag_std)
                    critic_distribution = Normal(
                        critic_mu.detach(), critic_std.detach()
                    )
                    msg_aggr_loss = kl_divergence(
                        ag_distribution, critic_distribution
                    ).mean()

                    # Use the predetermined loss function
                    if self.current_epoch < self.warmup_epochs:
                        critic_loss = td_error  # Only use TD error during warmup
                    else:
                        critic_loss = self.compute_critic_loss(td_error, msg_aggr_loss)

                    if self.use_ddp:
                        critic_loss = critic_loss.mean()  # Reduce across all GPUs

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
            self.eval_critic = self.eval_critic.module
            self.target_agent_group.unwrap_data_parallel()
            self.target_critic = self.target_critic.module

        self.eval_agent_group.to("cpu")
        self.eval_critic.to("cpu")
        self.target_agent_group.to("cpu")
        self.target_critic.to("cpu")
        torch.cuda.empty_cache()

        return total_loss / total_batches
