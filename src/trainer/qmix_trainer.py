import torch
from torch.nn import DataParallel
from tqdm import tqdm

from .trainer import Trainer
from ..util.trajectory_dataset import TrajectoryDataLoader

class QMIXTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def learn(self, sample_size, batch_size: int, times: int = 1):
        total_loss = 0.0
        total_batches = 0

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
                    observations = batch['observations']
                    states = batch['states']
                    actions = batch['actions']
                    rewards = batch['rewards']
                    next_states = batch['next_states']
                    next_observations = batch['next_observations']
                    terminations = batch['terminations']
                    bs = states.shape[0]  # Actual batch size
                    # Compute the Q-tot
                    self.eval_agent_group.train().to(self.train_device)
                    ret = self.eval_agent_group.forward(observations) # obs.shape (B, N, T, F)
                    q_val = ret['q_val']
                    actions = torch.Tensor(actions[:,:,-1:]).to(device=self.train_device, dtype=torch.int64) # (B, N, T, A)
                    q_val = torch.gather(q_val, dim=-1, index=actions)
                    q_val = q_val.squeeze(-1) # (B, N, 1) -> (B, N)
                    states = torch.Tensor(states[:,-1,:]).to(self.train_device) # (B, T, F) -> (B, F) Take only the last state in the sequence
                    self.eval_critic.train().to(self.train_device)
                    q_tot = self.eval_critic(q_val, states)

                    # Compute TD targets
                    with torch.no_grad():
                        self.target_agent_group.eval().to(self.train_device)
                        ret_next = self.eval_agent_group.forward(next_observations)
                        q_val_next = ret_next['q_val']
                        q_val_next = q_val_next.max(dim=-1).values
                        next_states = torch.Tensor(next_states[:,-1,:]).to(self.train_device) # (B, T, F) -> (B, F) Take only the last state in the sequence
                        self.target_critic.eval().to(self.train_device)
                        q_tot_next = self.target_critic(q_val_next, next_states)

                    # Compute the TD target
                    rewards = torch.Tensor(rewards[:,:,-1]).to(self.train_device) # (B, N, T) -> (B, N)
                    rewards = rewards.sum(dim=1) # (B, N) -> (B) Sum over all agents rewards
                    terminations = torch.Tensor(terminations[:,:,-1]).to(self.train_device) # (B, N, T) -> (B, N)
                    terminations = terminations.prod(dim=1) # (B, N) -> (B) if all agents are terminated then game over

                    y_tot = rewards + (1 - terminations) * self.gamma * q_tot_next

                    # Compute the critic loss
                    critic_loss = torch.nn.functional.mse_loss(q_tot, y_tot.detach())
                    if self.use_data_parallel:
                        critic_loss = critic_loss.mean() # Reduce across all GPUs

                    # Optimize the critic network
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.eval_critic.parameters(),
                        max_norm=5.0
                    )
                    self.optimizer.step()
                    self.eval_agent_group.step()

                    self.eval_agent_group.zero_grad()
                    self.eval_critic.zero_grad()

                    total_loss += critic_loss.detach().cpu().item()
                    total_batches += 1

                    pbar.update(bs)

        self.eval_agent_group.to("cpu")
        self.eval_critic.to("cpu")
        self.target_agent_group.to("cpu")
        self.target_critic.to("cpu")
        torch.cuda.empty_cache()

        if self.use_data_parallel:
            self.eval_agent_group.unwrap_data_parallel()
            self.eval_critic = self.eval_critic.module
            self.target_agent_group.unwrap_data_parallel()
            self.target_critic = self.target_critic.module

        return total_loss / total_batches