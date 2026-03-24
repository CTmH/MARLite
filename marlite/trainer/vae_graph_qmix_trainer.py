import torch
import numpy as np
import time
import absl.logging as logging
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from marlite.trainer.self_supervised_qmix_trainer import SelfSupervisedQMIXTrainer
from marlite.trainer.graph_qmix_trainer import GraphQMIXTrainer
from marlite.util.distributed_utils import get_local_device_id, average_loss


class VAEGraphQMIXTrainer(SelfSupervisedQMIXTrainer):
    """
    A GraphQMIXTrainer subclass that works with probabilistic agent groups.
    This trainer handles ProbObsGNNCommAgentGroup, ProbSeqGNNCommAgentGroup,
    DualPathObsGNNCommAgentGroup, and DualPathProbObsGNNCommAgentGroup.
    It optimizes the parameters of the independent multivariate Gaussian distributions
    output by probabilistic agent groups using a VAE decoder.
    """

    def __init__(self, kl_divergence_weight=1.0, **kwargs):

        super().__init__(**kwargs)
        self.kl_divergence_weight = kl_divergence_weight

    def _init_ssl_optimizer(self):
        # Create separate optimizers for self-supervised learning
        ssl_params_optim = [
            {"params": self.ssl_model.parameters()},
        ]
        self.ssl_optimizer = self.ssl_optimizer_config.get_optimizer(ssl_params_optim)
        return self

    def learn(self, sample_size, batch_size: int, times: int = 1):
        return GraphQMIXTrainer.learn(self, sample_size, batch_size, times)

    def self_supervised_learn(self, sample_size, batch_size: int, times: int = 1):
        total_loss = 0.0
        total_batches = 0

        # Check if DDP is enabled
        if self.use_ddp:
            # Multi-GPU DDP training
            device_id = get_local_device_id(self.train_device)

            # Move models to device and wrap with DDP
            self.eval_agent_group.wrap_data_parallel(device_id)
            self.ssl_model = DDP(
                self.ssl_model.to(self.train_device), device_ids=[device_id]
            )
        else:
            # Single device training
            self.eval_agent_group.to(self.train_device)
            self.ssl_model.to(self.train_device)

        # VAE self supervised learning
        self.eval_agent_group.train()
        for t in range(times):
            data = self.replaybuffer.sample(sample_size)
            data = list(data)
            alive_mask = np.stack([e["alive_mask"] for e in data])
            timestep_padding_mask = np.stack([e["timestep_padding_mask"] for e in data])
            observations = np.stack([e["observations"] for e in data]).astype(
                np.float32
            )
            states = np.stack([e["states"] for e in data])
            edge_indices = [e["edge_indices"] for e in data]

            start_time = time.time()
            formatted_obs, construct_padding_mask = self.data_constructor.process(
                observations, states, edge_indices, alive_mask
            )
            end_time = time.time()
            processing_time = end_time - start_time
            logging.info(
                f"Processing the self-supervised learning data takes {processing_time:.4f} seconds."
            )

            observations = torch.tensor(observations)
            formatted_obs = torch.tensor(formatted_obs)
            construct_padding_mask = torch.tensor(construct_padding_mask)
            timestep_padding_mask = torch.tensor(timestep_padding_mask)
            edge_indices_idx = torch.arange(len(edge_indices), dtype=torch.int)
            dataset = TensorDataset(
                observations,
                timestep_padding_mask,
                formatted_obs,
                edge_indices_idx,
                construct_padding_mask,
            )
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True, num_workers=self.n_workers
            )
            n_agents = alive_mask.shape[2]
            with tqdm(
                total=sample_size, desc=f"Times {t + 1}/{times}", unit="batch"
            ) as pbar:
                for obs, obs_mask, formatted, edge_idx, construct_mask in dataloader:
                    bs = obs.shape[0]
                    last_ts_edges = [edge_indices[i][-1] for i in edge_idx.tolist()]
                    obs = obs.to(self.train_device, dtype=torch.float32)
                    obs_mask = obs_mask.to(dtype=torch.bool)
                    obs_mask = torch.stack([obs_mask] * n_agents, dim=1).to(
                        self.train_device
                    )  # (B, N, T)
                    formatted = formatted.to(self.train_device, dtype=torch.float32)
                    construct_mask = construct_mask.to(
                        self.train_device, dtype=torch.bool
                    )
                    obs = obs.transpose(1, 2)  # (B, T, N, O) -> (B, N, T, O)
                    msg, _ = self.eval_agent_group._process_observations(obs, obs_mask)
                    estimates, _, mu, _, log_var = (
                        self.eval_agent_group._compute_local_state_estimates(
                            msg, last_ts_edges
                        )
                    )
                    reconstructed_obs = self.ssl_model(estimates)
                    reconstructed_obs = torch.reshape(
                        reconstructed_obs, formatted.shape
                    )
                    reconstruction_loss = self._compute_ssl_loss(
                        reconstructed_obs.view(-1, *reconstructed_obs.shape[2:]),
                        formatted.view(-1, *formatted.shape[2:]),
                        construct_mask.view(-1, *construct_mask.shape[2:]),
                    )
                    # Calculate KL divergence loss
                    # KL divergence between the learned distribution q(z|x) and prior p(z)
                    # For Gaussian prior N(0, I) and posterior N(mu, std^2), KL divergence is:
                    # If log_var is provided, use it directly for better numerical stability
                    # KL(q(z|x) || p(z)) = 0.5 * sum(1 + log_var - mu^2 - exp(log_var))

                    kl_divergence = -0.5 * torch.sum(
                        1 + log_var - mu.pow(2) - torch.exp(log_var), dim=-1
                    )  # Sum over feature dimension

                    kl_divergence = torch.mean(
                        kl_divergence
                    )  # Average over batch and agents

                    # Total VAE loss: reconstruction loss + KL divergence loss
                    vae_loss = (
                        reconstruction_loss + self.kl_divergence_weight * kl_divergence
                    )

                    # Optimize the networks - only use the main optimizer since it includes both critic and decoder params
                    self.eval_agent_group.zero_grad()
                    self.ssl_model.zero_grad()

                    vae_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.ssl_model.parameters()), max_norm=5.0
                    )
                    self.ssl_optimizer.step()
                    self.eval_agent_group.step()

                    total_loss += vae_loss.detach().cpu().item()
                    total_batches += 1

                    pbar.update(bs)

        if self.use_ddp:
            self.eval_agent_group.unwrap_data_parallel()
            self.ssl_model = self.ssl_model.module.cpu()
        else:
            self.eval_agent_group.to("cpu")
            self.ssl_model.to("cpu")

        torch.cuda.empty_cache()

        return total_loss / total_batches
