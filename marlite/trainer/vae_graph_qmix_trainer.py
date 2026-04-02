import torch
import numpy as np
import time
import absl.logging as logging
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from marlite.trainer.self_supervised_qmix_trainer import SelfSupervisedQMIXTrainer
from marlite.trainer.graph_qmix_trainer import GraphQMIXTrainer
from marlite.trainer.trainer_worker_group import GraphWorkerGroup, VAESSLWorkerGroup


class VAEGraphQMIXTrainer(SelfSupervisedQMIXTrainer):
    """
    A GraphQMIXTrainer subclass that works with probabilistic agent groups.
    This trainer handles ProbObsGNNCommAgentGroup, ProbSeqGNNCommAgentGroup,
    DualPathObsGNNCommAgentGroup, and DualPathProbObsGNNCommAgentGroup.
    It optimizes the parameters of the independent multivariate Gaussian distributions
    output by probabilistic agent groups using a VAE decoder.
    """

    def __init__(self, kl_divergence_weight=1.0, **kwargs):
        self.kl_divergence_weight = kl_divergence_weight
        super().__init__(**kwargs)

    def _create_worker_group(self):
        """Create GraphWorkerGroup for multi-GPU RL training."""
        if not self.use_multi_gpu:
            return None

        return GraphWorkerGroup(
            device_ids=list(range(len(self.device_list))),
            agent_group_config=self.agent_group_config,
            critic_config=self.critic_config,
            critic_optimizer_config=self.critic_optimizer_config,
            agent_optimizer_config=self.agent_optimizer_config,
            gamma=self.gamma,
        )

    def _create_ssl_worker_group(self):
        """Create VAESSLWorkerGroup for multi-GPU SSL training."""
        if not self.use_multi_gpu:
            return None

        return VAESSLWorkerGroup(
            device_ids=list(range(len(self.device_list))),
            ssl_model_config=self.ssl_model_config,
            agent_group_config=self.agent_group_config,
            ssl_optimizer_config=self.ssl_optimizer_config,
            agent_optimizer_config=self.agent_optimizer_config,
            reconstruction_loss=self.reconstruction_loss,
            kl_divergence_weight=self.kl_divergence_weight,
            data_constructor=self.data_constructor,
        )

    def learn(self, sample_size, batch_size: int, times: int = 1):
        """RL learning delegates to GraphQMIXTrainer's implementation."""
        if not self.use_multi_gpu:
            return GraphQMIXTrainer._learn_single_gpu(
                self, sample_size, batch_size, times
            )
        return GraphQMIXTrainer._learn_multi_gpu(self, sample_size, batch_size, times)

    def self_supervised_learn(self, sample_size, batch_size: int, times: int = 1):
        if not self.use_multi_gpu:
            return self._ssl_learn_single_gpu(sample_size, batch_size, times)
        return self._ssl_learn_multi_gpu(sample_size, batch_size, times)

    def _ssl_learn_single_gpu(self, sample_size, batch_size: int, times: int = 1):
        """Single GPU SSL learning."""
        total_loss = 0.0
        total_batches = 0

        self.eval_agent_group.to(self.train_device)
        self.ssl_model.to(self.train_device)

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
                    )
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

                    kl_divergence = -0.5 * torch.sum(
                        1 + log_var - mu.pow(2) - torch.exp(log_var), dim=-1
                    )
                    kl_divergence = torch.mean(kl_divergence)

                    vae_loss = (
                        reconstruction_loss + self.kl_divergence_weight * kl_divergence
                    )

                    self.agent_optimizer.zero_grad()
                    self.ssl_model.zero_grad()

                    vae_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.ssl_model.parameters()), max_norm=5.0
                    )
                    torch.nn.utils.clip_grad_norm_(
                        self.eval_agent_group.parameters(), max_norm=5.0
                    )
                    self.ssl_optimizer.step()
                    self.agent_optimizer.step()

                    total_loss += vae_loss.detach().cpu().item()
                    total_batches += 1

                    pbar.update(bs)

        self.eval_agent_group.to("cpu")
        self.ssl_model.to("cpu")

        torch.cuda.empty_cache()

        return total_loss / total_batches

    def _ssl_learn_multi_gpu(self, sample_size, batch_size: int, times: int = 1):
        """Multi-GPU SSL learning via worker processes."""
        self.ssl_worker_group.move_models_to_gpu()

        total_loss = 0.0
        total_batches = 0

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
                total=sample_size, desc=f"SSL Times {t + 1}/{times}", unit="batch"
            ) as pbar:
                for obs, obs_mask, formatted, edge_idx, construct_mask in dataloader:
                    batch = {
                        "observations": obs,
                        "timestep_padding_mask": obs_mask,
                        "formatted_obs": formatted,
                        "construct_padding_mask": construct_mask,
                        "edge_indices": [edge_indices[i] for i in edge_idx.tolist()],
                        "n_agents": n_agents,
                        "epoch": self.current_epoch,
                    }

                    loss = self.ssl_worker_group.ssl_train_step(batch)

                    total_loss += loss
                    total_batches += 1

                    bs = obs.shape[0]
                    pbar.update(bs)

        self.ssl_worker_group.move_models_to_cpu()
        torch.cuda.empty_cache()

        return total_loss / total_batches
