import torch
from torch.nn import DataParallel
from tqdm import tqdm

from marlite.trainer.trainer import Trainer
from marlite.util.trajectory_dataset import TrajectoryDataLoader

class QMIXTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
                    states = batch['states']
                    actions = batch['actions']
                    rewards = batch['rewards']
                    next_states = batch['next_states']
                    next_observations = batch['next_observations']
                    terminations = batch['terminations']
                    truncations = batch['truncations']
                    bs = states.shape[0]  # Actual batch size

                    terminations = torch.tensor(terminations[:,:,-1]).to(self.train_device) # (B, N, T) -> (B, N)
                    truncations = torch.tensor(truncations[:,:,-1]).to(self.train_device) # (B, N, T) -> (B, N)
                    alive_mask_next = ~(terminations | truncations)
                    alive_mask = torch.tensor(alive_mask[:,:,-1]).to(self.train_device) # (B, N, T) -> (B, N)

                    rewards = torch.Tensor(rewards[:,:,-1]).to(self.train_device) # (B, N, T) -> (B, N)
                    rewards = rewards.sum(dim=1) # (B, N) -> (B) Sum over all agents rewards
                    terminations = terminations.prod(dim=1) # (B, N) -> (B) if all agents are terminated then game over

                    # Compute the Q-tot
                    self.eval_agent_group.train()
                    ret = self.eval_agent_group.forward(observations) # obs.shape (B, N, T, F)
                    q_val = ret['q_val']
                    actions = torch.Tensor(actions[:,:,-1:]).to(device=self.train_device, dtype=torch.int64) # (B, N, T, A) -> (B, N, A)
                    q_val = torch.gather(q_val, dim=-1, index=actions)
                    q_val = q_val.squeeze(-1) # (B, N, 1) -> (B, N)
                    q_val = q_val * alive_mask
                    states = torch.Tensor(states[:,-1,:]).to(self.train_device) # (B, T, F) -> (B, F) Take only the last state in the sequence
                    self.eval_critic.train()
                    ret = self.eval_critic(q_val, states)
                    q_tot = ret['q_tot']

                    # Compute TD targets
                    with torch.no_grad():
                        self.target_agent_group.eval()
                        ret_next = self.eval_agent_group.forward(next_observations)
                        q_val_next = ret_next['q_val']
                        q_val_next = q_val_next.max(dim=-1).values
                        q_val_next = q_val_next * alive_mask_next
                        next_states = torch.Tensor(next_states[:,-1,:]).to(self.train_device) # (B, T, F) -> (B, F) Take only the last state in the sequence
                        self.target_critic.eval()
                        ret_next = self.target_critic(q_val_next, next_states)
                        q_tot_next = ret_next['q_tot']

                    # Compute the TD target
                    y_tot = rewards + (1 - terminations) * self.gamma * q_tot_next

                    # Compute the critic loss
                    critic_loss = torch.nn.functional.mse_loss(q_tot, y_tot.detach())
                    if self.use_data_parallel:
                        critic_loss = critic_loss.mean() # Reduce across all GPUs

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