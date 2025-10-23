import numpy as np
import torch
from torch.nn import DataParallel
from tqdm import tqdm

from marlite.trainer.trainer import Trainer
from marlite.util.trajectory_dataset import TrajectoryDataLoader
from marlite.util.loss_func import PITLoss

class MsgAggrQMIXTrainer(Trainer):
    def __init__(self, **kwargs):
        margin = kwargs.pop('triplet_loss_margin', 1.0)
        pit_loss_alpha = kwargs.pop('pit_loss_alpha', 0.9)
        super().__init__(**kwargs)
        self.triplet_loss = torch.nn.TripletMarginLoss(margin=margin)
        self.pit_loss = PITLoss(num_tasks=2, alpha=pit_loss_alpha)

    def learn(self, sample_size, batch_size: int, times: int = 1):
        total_loss = 0.0
        total_batches = 0

        # Move models to the appropriate device before wrapping with DataParallel
        self.eval_agent_group.to(self.train_device)
        self.eval_critic.to(self.train_device)
        self.target_agent_group.to(self.train_device)
        self.target_critic.to(self.train_device)

        if self.use_data_parallel:
            self.eval_agent_group.wrap_data_parallel()
            self.eval_critic = DataParallel(self.eval_critic)
            self.target_agent_group.wrap_data_parallel()
            self.target_critic = DataParallel(self.target_critic)

        for t in range(times):
            with tqdm(total=sample_size, desc=f'Times {t+1}/{times}', unit='batch') as pbar:
                # Implement the learning logic for QMix
                # Get a batch of data from the replay buffer
                dataset = self.replaybuffer.sample(sample_size)
                dataloader = TrajectoryDataLoader(dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=self.n_workers)
                for batch in dataloader:
                    # Extract batch data
                    alive_mask = batch['alive_mask']
                    observations = batch['observations']
                    obs_padding_mask = batch['obs_padding_mask']
                    states = batch['states']
                    actions = batch['actions']
                    rewards = batch['rewards']
                    next_states = batch['next_states']
                    next_observations = batch['next_observations']
                    next_obs_padding_mask = batch['next_obs_padding_mask']
                    next_avail_actions = batch['next_avail_actions']
                    terminations = batch['terminations']
                    truncations = batch['truncations']
                    bs = states.shape[0]  # Actual batch size
                    n_agents = rewards.shape[1]

                    # Create alive_mask_next from terminations and truncations
                    alive_mask = torch.tensor(alive_mask).to(dtype=torch.bool) # (B, N, T)
                    alive_mask = alive_mask.permute(0, 2, 1).to(self.train_device) # (B, N, T) -> (B, T, N)
                    terminations = torch.tensor(terminations[:,:,-1]).to(self.train_device) # (B, N, T) -> (B, N)
                    truncations = torch.tensor(truncations[:,:,-1]).to(self.train_device) # (B, N, T) -> (B, N)
                    next_alive_mask = ~(terminations | truncations)
                    next_alive_mask = next_alive_mask.unsqueeze(dim=1)
                    next_alive_mask = torch.cat([alive_mask[:,1:,:], next_alive_mask], dim=1)
                    # Action mask: (B, N, T, Actions) -> (B, N, Actions)
                    if np.issubdtype(next_avail_actions.dtype, np.number):
                        use_action_mask = True
                        next_avail_actions = torch.tensor(next_avail_actions[:,:,-1,:])
                        next_avail_actions = next_avail_actions.to(dtype=torch.bool, device=self.train_device)
                    else:
                        use_action_mask = False

                    rewards = torch.Tensor(rewards[:,:,-1]).to(self.train_device) # (B, N, T) -> (B, N)
                    rewards = rewards.sum(dim=1) # (B, N) -> (B) Sum over all agents rewards
                    terminations = terminations.prod(dim=1) # (B, N) -> (B) if all agents are terminated then game over

                    obs_padding_mask = torch.tensor(obs_padding_mask, dtype=torch.bool) # (B, T)
                    obs_padding_mask = torch.stack([obs_padding_mask] * n_agents, dim=1).to(self.train_device)  # (B, N, T)
                    next_obs_padding_mask = torch.tensor(next_obs_padding_mask, dtype=torch.bool)
                    next_obs_padding_mask = torch.stack([next_obs_padding_mask] * n_agents, dim=1).to(self.train_device)

                    # Compute the Q-tot
                    self.eval_agent_group.train()
                    observations = torch.tensor(observations, dtype=torch.float, device=self.train_device)
                    ret = self.eval_agent_group.forward(observations, obs_padding_mask, alive_mask[:,-1,:]) # obs.shape (B, N, T, F)
                    q_val = ret['q_val']
                    aggregated_msg = ret['aggregated_msg']
                    actions = torch.Tensor(actions[:,:,-1:]).to(device=self.train_device, dtype=torch.int64) # (B, N, T, A)
                    q_val = torch.gather(q_val, dim=-1, index=actions)
                    q_val = q_val.squeeze(-1) # (B, N, 1) -> (B, N)
                    states = torch.Tensor(states).to(self.train_device) # (B, T, F)
                    self.eval_critic.train()
                    ret = self.eval_critic(q_val, states, alive_mask, obs_padding_mask[:,0,:])
                    q_tot = ret['q_tot']
                    state_features = ret['state_features']

                    # Compute TD targets
                    with torch.no_grad():
                        self.target_agent_group.eval()
                        next_observations = torch.tensor(next_observations, dtype=torch.float, device=self.train_device)
                        ret_next = self.eval_agent_group.forward(next_observations, next_obs_padding_mask, next_alive_mask[:,-1,:])
                        q_val_next = ret_next['q_val']
                        #aggregated_msg_next = ret_next['aggregated_msg']
                        if use_action_mask:
                            q_val_next = torch.masked_fill(q_val_next, ~next_avail_actions, -torch.inf)
                        q_val_next = q_val_next.max(dim=-1).values
                        next_states = torch.Tensor(next_states).to(self.train_device) # (B, T, F)
                        self.target_critic.eval()
                        ret_next = self.target_critic(q_val_next, next_states, next_alive_mask, next_obs_padding_mask[:,0,:])
                        q_tot_next = ret_next['q_tot']
                        #state_features_next = ret_next['state_features']

                    # Compute the TD target
                    y_tot = rewards + (1 - terminations) * self.gamma * q_tot_next

                    # TD error
                    td_error = torch.nn.functional.mse_loss(q_tot, y_tot.detach())
                    # Message aggregation loss
                    indices = torch.randperm(bs).to(aggregated_msg.device)
                    negatives = aggregated_msg[indices]
                    msg_aggr_loss = self.triplet_loss(aggregated_msg, state_features.detach(), negatives)

                    self.pit_loss.to(self.train_device)
                    critic_loss = self.pit_loss(torch.stack([td_error, msg_aggr_loss]))
                    if self.use_data_parallel:
                        critic_loss = torch.mean(critic_loss) # Reduce across all GPUs

                    # Optimize the critic network
                    self.eval_agent_group.zero_grad()
                    self.eval_critic.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.eval_critic.parameters(),
                        max_norm=5.0
                    )
                    self.optimizer.step()
                    self.eval_agent_group.step()

                    total_loss += critic_loss.detach().cpu().item()
                    total_batches += 1

                    pbar.update(bs)

        if self.use_data_parallel:
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