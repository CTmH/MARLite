"""
SSL Group Consensus MAPPO worker for multi-GPU training.
"""

import io
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributions import Categorical
import numpy as np
from typing import Any, Dict

from marlite.algorithm.agents import AgentGroupConfig
from marlite.algorithm.critic import CriticConfig
from marlite.algorithm.model import ModelConfig
from marlite.util.optimizer_config import OptimizerConfig
from marlite.util.loss_func import PITLoss, ReconstructionLoss
from marlite.trainer.trainer_worker.onpolicy_worker import OnPolicyWorker


class SSLGroupConsensusMAPPOWorker(OnPolicyWorker):
    critic_optimizer: torch.optim.Optimizer
    agent_optimizer: torch.optim.Optimizer
    ssl_optimizer: torch.optim.Optimizer

    def __init__(
        self,
        worker_id: int,
        device_id: int,
        rank: int,
        world_size: int,
        init_method: str,
        agent_group_config: AgentGroupConfig,
        critic_config: CriticConfig,
        critic_optimizer_config: OptimizerConfig,
        agent_optimizer_config: OptimizerConfig,
        ssl_model_config: ModelConfig,
        ssl_optimizer_config: OptimizerConfig,
        reconstruction_loss,
        data_constructor,
        gamma: float,
        max_grad_norm: float,
        clip_epsilon: float,
        gae_lambda: float,
        entropy_coef: float,
        vf_coef: float,
        kl_divergence_weight: float,
        self_supervised_learning_loss_weight: float,
        loss_combination_method: str,
        pit_loss_alpha: float,
        warmup_iterations: int,
        recon_mode: str,
        kl_on_agent: bool,
        kl_on_group: bool,
        consensus_mode: str,
        **kwargs,
    ):
        super().__init__(worker_id, device_id, rank, world_size, init_method)
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.clip_epsilon = clip_epsilon
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.kl_divergence_weight = kl_divergence_weight
        self.recon_mode = recon_mode
        self.kl_on_agent = kl_on_agent
        self.kl_on_group = kl_on_group
        self.warmup_iterations = warmup_iterations
        self.consensus_mode = consensus_mode
        self.self_supervised_learning_loss_weight = self_supervised_learning_loss_weight
        self.loss_combination_method = loss_combination_method

        self.eval_agent_group = agent_group_config.get_agent_group()
        self.eval_critic = critic_config.get_critic()

        self.eval_agent_group.train()
        self.eval_critic.train()

        self.critic_optimizer = critic_optimizer_config.get_optimizer(
            self.eval_critic.parameters()
        )
        self.agent_optimizer = agent_optimizer_config.get_optimizer(
            self.eval_agent_group.parameters()
        )

        self.data_constructor = data_constructor

        self.ssl_model = ssl_model_config.get_model()
        self.reconstruction_loss = reconstruction_loss
        if not isinstance(self.reconstruction_loss, ReconstructionLoss):
            raise TypeError(
                f"reconstruction_loss must be a ReconstructionLoss subclass, "
                f"got {type(self.reconstruction_loss).__name__}"
            )
        self.ssl_optimizer = ssl_optimizer_config.get_optimizer(
            self.ssl_model.parameters()
        )
        self.pit_loss = PITLoss(
            num_tasks=2, alpha=pit_loss_alpha, reduction="mean"
        )

    def move_to_device(self, device: str):
        if self.eval_agent_group is not None:
            self.eval_agent_group.to(device)
        if self.eval_critic is not None:
            self.eval_critic.to(device)
        self.ssl_model.to(device)
        self.device = device

    def reduce_gradients(self):
        super().reduce_gradients()
        for param in self.ssl_model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.world_size

    def synchronize_eval_params(self):
        super().synchronize_eval_params()
        for param in self.ssl_model.parameters():
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data /= self.world_size

    def get_params_for_main(self) -> Dict[str, Any]:
        params = {
            "eval_agent_group": {
                k: v.clone().cpu()
                for k, v in self.eval_agent_group.state_dict().items()
            },
            "eval_critic": {
                k: v.clone().cpu() for k, v in self.eval_critic.state_dict().items()
            },
            "ssl_model": {
                k: v.clone().cpu() for k, v in self.ssl_model.state_dict().items()
            },
        }
        return params

    def sync_params_from_main(self, params):
        # Delegate to OnPolicyWorker for bytes-deserialisation and
        # standard eval_agent_group + eval_critic handling.  SSL
        # auxiliary model is added here.
        params = super().sync_params_from_main(params)
        if "ssl_model" in params and self.ssl_model is not None:
            self.ssl_model.load_state_dict(
                {k: v.clone() for k, v in params["ssl_model"].items()}
            )

    # ── SSL helpers ──────────────────────────────────────────────────────

    def _compute_ssl_loss(self, pred_set, target_set, mask=None):
        return self.reconstruction_loss(pred_set, target_set, mask)

    def _combine_rl_ssl_loss(self, rl_loss, ssl_loss):
        if self.loss_combination_method == "pit_loss":
            return self.pit_loss(torch.stack([rl_loss, ssl_loss]))
        return rl_loss + self.self_supervised_learning_loss_weight * ssl_loss

    def _build_recon_targets(self, observations, states, group_indices, alive_mask):
        obs_np = observations.detach().cpu().numpy()
        states_np = states.detach().cpu().numpy()
        alive_np = alive_mask.detach().cpu().numpy()

        targets_np, construct_mask_np = self.data_constructor.process(
            observations=obs_np, states=states_np,
            grouping=group_indices, alive_mask=alive_np,
        )

        targets = torch.tensor(targets_np, dtype=torch.float32, device=self.device)
        construct_mask = torch.tensor(construct_mask_np, dtype=torch.bool, device=self.device)
        return targets, construct_mask

    def _recon_loss_per_agent(self, consensus, group_indices, targets, construct_mask, alive_mask):
        bs, G, L = consensus.shape
        n_agents = group_indices.shape[1]
        device = consensus.device
        target_flat_dim = targets.shape[2:]

        pred_flat = self.ssl_model(consensus.reshape(bs * G, L))
        pred_g = pred_flat.reshape(bs, G, *target_flat_dim)

        gids = torch.as_tensor(group_indices, device=device)
        dead = gids < 0
        mask = F.one_hot(gids.clamp(min=0), num_classes=G).float()
        mask[dead] = 0.0
        mask = mask * construct_mask.unsqueeze(1).float()

        flat_pred = pred_g.reshape(bs, G, -1)
        flat_target = targets.reshape(bs, G, -1)
        pred_agent = torch.bmm(mask, flat_pred).reshape(bs, n_agents, *target_flat_dim)
        target_agent = torch.bmm(mask, flat_target).reshape(bs, n_agents, *target_flat_dim)

        agent_alive = alive_mask[:, -1, :]
        return self._compute_ssl_loss(pred_agent, target_agent, agent_alive)

    def _recon_loss_per_group(self, consensus, targets, construct_mask):
        bs, G, L = consensus.shape
        target_flat_dim = targets.shape[2:]

        pred_flat = self.ssl_model(consensus.reshape(bs * G, L))
        pred_g = pred_flat.reshape(bs, G, *target_flat_dim)

        return self._compute_ssl_loss(pred_g, targets, construct_mask)

    # ── training step ────────────────────────────────────────────────────

    def train_step(self, batch: Dict[str, Any]) -> tuple:
        current_epoch = batch.get("epoch", 0)
        is_warmup = current_epoch < self.warmup_iterations

        alive_mask = batch["alive_mask"].to(dtype=torch.bool)
        observations = batch["observations"].to(dtype=torch.float32)
        timestep_padding_mask = batch["timestep_padding_mask"].to(
            dtype=torch.bool, device=self.device
        )
        states = batch["states"].to(dtype=torch.float32)
        actions = batch["actions"].to(dtype=torch.int)
        rewards = batch["rewards"].to(dtype=torch.float32)
        next_states = batch["next_states"].to(dtype=torch.float32)
        next_timestep_padding_mask = batch["next_timestep_padding_mask"].to(
            dtype=torch.bool, device=self.device
        )
        next_alive_mask = batch["next_alive_mask"].to(dtype=torch.bool)
        all_log_probs = batch["all_log_probs"].to(dtype=torch.float32)
        terminations = batch["terminations"].to(dtype=torch.bool)

        bs = states.shape[0]
        n_agents = rewards.shape[2]
        t_steps = rewards.shape[1]

        alive_mask = alive_mask.to(self.device)
        next_alive_mask = next_alive_mask.to(self.device)
        states_dev = states.to(self.device)
        next_states_dev = next_states.to(self.device)

        group_indices_batch = batch.get("group_indices")
        if group_indices_batch is not None:
            group_indices_np = group_indices_batch[:, -1, :].numpy()
        else:
            group_indices_np = np.zeros((bs, n_agents), dtype=np.int64) - 1

        # ── critic forward ──
        self.eval_critic.train()
        v = self.eval_critic(states_dev, alive_mask, timestep_padding_mask)["v"]
        v_last = v[:, 0]

        with torch.no_grad():
            v_next = self.eval_critic(
                next_states_dev[:, -1:, ...],
                next_alive_mask[:, -1:, ...],
                next_timestep_padding_mask[:, -1:],
            )["v"][:, 0]

        r_last = self._aggregate_rewards(rewards[:, -1]).to(self.device)
        termination_last = terminations[:, -1].prod(dim=-1).to(
            dtype=torch.float32, device=self.device
        )
        delta = r_last + self.gamma * v_next * (1.0 - termination_last) - v_last
        advantages_last = delta
        returns = delta + v_last

        # ── agent forward ──
        states_last = states_dev[:, -1]
        timestep_padding_mask_expanded = torch.stack(
            [timestep_padding_mask] * n_agents, dim=1
        ).to(self.device)
        observations_transposed = torch.transpose(observations, 1, 2).to(self.device)

        self.eval_agent_group.reset().train()
        ret_agent = self.eval_agent_group(
            observations_transposed, states_last,
            timestep_padding_mask_expanded, alive_mask[:, -1, :],
            group_indices_np,
        )
        action_logits = ret_agent["action_logits"]
        group_consensus = ret_agent.get("group_consensus")
        group_mu = ret_agent.get("group_mu")
        group_log_var = ret_agent.get("group_log_var")
        agent_mu = ret_agent.get("agent_mu")
        agent_log_var = ret_agent.get("agent_log_var")

        # ── PPO actor loss ──
        actions_last = actions[:, -1].to(dtype=torch.int64, device=self.device)
        log_probs_old = all_log_probs[:, -1, :].to(self.device)

        dist = Categorical(logits=action_logits)
        new_log_probs = dist.log_prob(actions_last)
        entropy = dist.entropy()

        alive_last_flag = alive_mask[:, -1, :].to(
            dtype=torch.float32, device=self.device
        )
        alive_last_count = alive_last_flag.sum()

        ratio = torch.exp(new_log_probs - log_probs_old)
        adv_expanded = advantages_last.unsqueeze(-1).expand(-1, n_agents)
        surr1 = ratio * adv_expanded
        surr2 = (
            torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
            * adv_expanded
        )
        actor_loss = (
            -(torch.min(surr1, surr2) * alive_last_flag).sum()
            / max(alive_last_count, torch.tensor(1.0, device=self.device))
        )
        entropy_loss = (
            -(entropy * alive_last_flag).sum()
            / max(alive_last_count, torch.tensor(1.0, device=self.device))
        )
        actor_loss = actor_loss + self.entropy_coef * entropy_loss

        # ── PPO critic loss ──
        critic_loss = F.mse_loss(v_last, returns.detach())

        # ── SSL reconstruction loss ──
        if is_warmup:
            ssl_loss = torch.tensor(0.0, device=self.device)
        else:
            # Pre-generated targets are provided by the trainer via
            # SSLEnrichedTrajectoryDataset — no per-batch GPU↔CPU round-trip.
            targets = batch["formatted_obs"].to(
                dtype=torch.float32, device=self.device
            )
            construct_mask = batch["construct_padding_mask"].to(
                dtype=torch.bool, device=self.device
            )
            if self.recon_mode == "per_group":
                recon_loss = self._recon_loss_per_group(
                    group_consensus, targets, construct_mask
                )
            else:
                recon_loss = self._recon_loss_per_agent(
                    group_consensus, group_indices_np, targets,
                    construct_mask, alive_mask,
                )

            if self.consensus_mode == "ae":
                kl_divergence = torch.tensor(0.0, device=self.device)
            else:
                kl_divergence = 0.0
                if self.kl_on_agent:
                    kl_mu = agent_mu
                    kl_log_var = agent_log_var
                    mask = alive_mask[:, -1, :].unsqueeze(-1).expand_as(agent_mu)
                    kl_per_dim = 1 + kl_log_var - kl_mu.pow(2) - torch.exp(kl_log_var)
                    kl_divergence = kl_divergence + -0.5 * (
                        kl_per_dim * mask
                    ).sum() / mask.sum().clamp(min=1)
                if self.kl_on_group:
                    kl_mu = group_mu
                    kl_log_var = group_log_var
                    mask = construct_mask.unsqueeze(-1).expand_as(group_mu)
                    kl_per_dim = 1 + kl_log_var - kl_mu.pow(2) - torch.exp(kl_log_var)
                    kl_divergence = kl_divergence + -0.5 * (
                        kl_per_dim * mask
                    ).sum() / mask.sum().clamp(min=1)

            ssl_loss = recon_loss + self.kl_divergence_weight * kl_divergence

        # ── combined loss ──
        rl_loss = actor_loss + self.vf_coef * critic_loss
        if is_warmup:
            combined_loss = rl_loss
        else:
            combined_loss = self._combine_rl_ssl_loss(rl_loss, ssl_loss)

        # ── backward ──
        self.critic_optimizer.zero_grad()
        self.agent_optimizer.zero_grad()
        self.ssl_optimizer.zero_grad()

        combined_loss.backward()

        self.reduce_gradients()

        torch.nn.utils.clip_grad_norm_(
            self.eval_critic.parameters(), max_norm=self.max_grad_norm
        )
        torch.nn.utils.clip_grad_norm_(
            self.eval_agent_group.parameters(), max_norm=self.max_grad_norm
        )
        if not is_warmup:
            torch.nn.utils.clip_grad_norm_(
                self.ssl_model.parameters(), max_norm=self.max_grad_norm
            )

        self.critic_optimizer.step()
        self.agent_optimizer.step()
        if not is_warmup:
            self.ssl_optimizer.step()

        ssl_loss_value = (
            ssl_loss.detach().cpu().item()
            if isinstance(ssl_loss, torch.Tensor)
            else ssl_loss
        )
        return (
            combined_loss.detach().cpu().item(),
            critic_loss.detach().cpu().item(),
            ssl_loss_value,
        )

    # ── handle_command (tuple-aware) ─────────────────────────────────────

    def handle_command(
        self,
        cmd: str,
        param_queue,
        data_queue,
        loss_queue,
        ack_queue=None,
    ) -> bool:
        if cmd == "STOP":
            self.cleanup()
            return False

        elif cmd == "SYNC_FROM_MAIN":
            params = param_queue.get()
            self.sync_params_from_main(params)
            del params
            ack_queue.put("ACK")

        elif cmd == "BROADCAST":
            params = param_queue.get()
            self.sync_params_from_main(params)
            del params

        elif cmd == "SYNC_TO_MAIN":
            params = self.get_params_for_main()
            param_queue.put(params)

        elif cmd == "TRAIN_STEP":
            batch = data_queue.get()
            result = self.train_step(batch)
            del batch
            combined, critic, ssl = result
            loss_queue.put((combined, critic, ssl))

        elif cmd == "MOVE_TO_GPU":
            self.move_to_device(self.assigned_device)
            if ack_queue:
                ack_queue.put("ACK")

        elif cmd == "MOVE_TO_CPU":
            self.move_to_device("cpu")
            torch.cuda.empty_cache()
            if ack_queue:
                ack_queue.put("ACK")

        elif cmd == "SYNC_LR":
            lr_data = param_queue.get()
            if "critic_lr" in lr_data:
                for param_group in self.critic_optimizer.param_groups:
                    param_group["lr"] = lr_data["critic_lr"]
            if "agent_lr" in lr_data:
                for param_group in self.agent_optimizer.param_groups:
                    param_group["lr"] = lr_data["agent_lr"]
            if "ssl_lr" in lr_data:
                for param_group in self.ssl_optimizer.param_groups:
                    param_group["lr"] = lr_data["ssl_lr"]
            if ack_queue:
                ack_queue.put("ACK")

        else:
            print(f"Worker {self.worker_id}: Unknown command: {repr(cmd)}", flush=True)

        return True

    def cleanup(self):
        if dist.is_initialized():
            dist.destroy_process_group()
