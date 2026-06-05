import io
import torch
import torch.nn.functional as F
import torch.distributed as dist
from typing import Any, Dict

from marlite.algorithm.agents import AgentGroupConfig
from marlite.algorithm.critic import CriticConfig
from marlite.algorithm.model import ModelConfig
from marlite.util.optimizer_config import OptimizerConfig
from marlite.util.loss_func import PITLoss
from marlite.trainer.trainer_worker.offpolicy_worker import OffPolicyWorker


class SSLGroupConsensusWorker(OffPolicyWorker):
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
        gamma: float = 0.9,
        max_grad_norm: float = 5.0,
        ssl_model_config: ModelConfig = None,
        ssl_optimizer_config: OptimizerConfig = None,
        reconstruction_loss=None,
        data_constructor=None,
        kl_divergence_weight: float = 0.005,
        self_supervised_learning_loss_weight: float = 1.0,
        loss_combination_method: str = "weighted_sum",
        pit_loss_alpha: float = 0.9,
        warmup_epochs: int = 0,
        recon_mode: str = "per_agent",
        kl_on_group: bool = False,
        kl_on_agent: bool = True,
        consensus_mode: str = "vae",
        **kwargs,
    ):
        if recon_mode not in ("per_agent", "per_group"):
            raise ValueError(
                f"recon_mode must be 'per_agent' or 'per_group', got '{recon_mode}'"
            )
        self.recon_mode = recon_mode
        self.kl_on_group = kl_on_group
        self.kl_on_agent = kl_on_agent
        self.warmup_epochs = warmup_epochs
        self.pit_loss_alpha = pit_loss_alpha
        self.kl_divergence_weight = kl_divergence_weight
        self.loss_combination_method = loss_combination_method
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.consensus_mode = consensus_mode

        super().__init__(worker_id, device_id, rank, world_size, init_method)

        self.eval_agent_group = agent_group_config.get_agent_group()
        self.target_agent_group = agent_group_config.get_agent_group()
        self.eval_critic = critic_config.get_critic()
        self.target_critic = critic_config.get_critic()

        self.eval_agent_group.train()
        self.target_agent_group.eval()
        self.eval_critic.train()
        self.target_critic.eval()

        self.critic_optimizer = critic_optimizer_config.get_optimizer(
            self.eval_critic.parameters()
        )
        self.agent_optimizer = agent_optimizer_config.get_optimizer(
            self.eval_agent_group.parameters()
        )

        self.self_supervised_learning_loss_weight = self_supervised_learning_loss_weight
        self.data_constructor = data_constructor

        self.ssl_model = ssl_model_config.get_model()
        self.reconstruction_loss = reconstruction_loss
        self.ssl_optimizer = ssl_optimizer_config.get_optimizer(
            self.ssl_model.parameters()
        )
        self.pit_loss = PITLoss(
            num_tasks=2,
            alpha=self.pit_loss_alpha,
            reduction="mean",
        )

    def move_to_device(self, device: str):
        if self.eval_agent_group is not None:
            self.eval_agent_group.to(device)
        if self.target_agent_group is not None:
            self.target_agent_group.to(device)
        if self.eval_critic is not None:
            self.eval_critic.to(device)
        if self.target_critic is not None:
            self.target_critic.to(device)
        self.ssl_model.to(device)
        self.device = device

    def reduce_gradients(self):
        for param in self.eval_critic.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.world_size

        for param in self.eval_agent_group.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.world_size

        for param in self.ssl_model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.world_size

    def get_params_for_main(self) -> Dict[str, Any]:
        params = {
            "eval_agent_group": {
                k: v.clone().cpu()
                for k, v in self.eval_agent_group.state_dict().items()
            },
            "target_agent_group": {
                k: v.clone().cpu()
                for k, v in self.target_agent_group.state_dict().items()
            },
            "eval_critic": {
                k: v.clone().cpu() for k, v in self.eval_critic.state_dict().items()
            },
            "target_critic": {
                k: v.clone().cpu() for k, v in self.target_critic.state_dict().items()
            },
            "ssl_model": {
                k: v.clone().cpu() for k, v in self.ssl_model.state_dict().items()
            },
        }
        return params

    def sync_params_from_main(self, params):
        if isinstance(params, bytes):
            buffer = io.BytesIO(params)
            params = torch.load(buffer, weights_only=True)

        if "eval_agent_group" in params and self.eval_agent_group is not None:
            self.eval_agent_group.load_state_dict(
                {k: v.clone() for k, v in params["eval_agent_group"].items()}
            )
        if "target_agent_group" in params and self.target_agent_group is not None:
            self.target_agent_group.load_state_dict(
                {k: v.clone() for k, v in params["target_agent_group"].items()}
            )
        if "eval_critic" in params and self.eval_critic is not None:
            self.eval_critic.load_state_dict(
                {k: v.clone() for k, v in params["eval_critic"].items()}
            )
        if "target_critic" in params and self.target_critic is not None:
            self.target_critic.load_state_dict(
                {k: v.clone() for k, v in params["target_critic"].items()}
            )
        self.ssl_model.load_state_dict(
            {k: v.clone() for k, v in params["ssl_model"].items()}
        )

    def train_step(self, batch: Dict[str, Any]) -> tuple:
        current_epoch = batch.get("epoch", 0)
        is_warmup = current_epoch < self.warmup_epochs

        alive_mask = batch["alive_mask"].to(dtype=torch.bool)
        observations = batch["observations"].to(dtype=torch.float32)
        timestep_padding_mask = batch["timestep_padding_mask"].to(dtype=torch.bool)
        states = batch["states"].to(dtype=torch.float32)
        actions = batch["actions"].to(dtype=torch.int)
        rewards = batch["rewards"].to(dtype=torch.float32)
        next_states = batch["next_states"].to(dtype=torch.float32)
        next_observations = batch["next_observations"].to(dtype=torch.float32)
        next_timestep_padding_mask = batch["next_timestep_padding_mask"].to(
            dtype=torch.bool
        )
        next_avail_actions = batch["next_avail_actions"]
        next_alive_mask = batch["next_alive_mask"].to(dtype=torch.bool)
        terminations = batch["terminations"].to(dtype=torch.bool)

        bs = states.shape[0]
        n_agents = rewards.shape[2]

        next_alive_mask = next_alive_mask.to(self.device)
        alive_mask = alive_mask.to(self.device)
        states = states.to(self.device)
        next_states = next_states.to(self.device)

        if isinstance(next_avail_actions, torch.Tensor):
            use_action_mask = True
            next_avail_actions = next_avail_actions[:, -1, :, :]
            next_avail_actions = next_avail_actions.to(
                dtype=torch.bool, device=self.device
            )
        else:
            use_action_mask = False

        rewards = rewards[:, -1]
        rewards = rewards.sum(dim=1).to(self.device)
        terminations = terminations[:, -1]
        terminations = terminations.prod(dim=1).to(self.device)

        timestep_padding_mask = torch.stack(
            [timestep_padding_mask] * n_agents, dim=1
        ).to(self.device)
        next_timestep_padding_mask = torch.stack(
            [next_timestep_padding_mask] * n_agents, dim=1
        ).to(self.device)

        states_last_np = states[:, -1].detach().cpu().numpy()
        group_indices = batch["group_indices"][:, -1, :].numpy()

        # === RL Forward Pass (Agent + standard QMixer) ===
        self.eval_agent_group.reset().train()
        observations_t = torch.transpose(observations, 1, 2).to(self.device)

        ret = self.eval_agent_group(
            observations_t,
            states_last_np,
            timestep_padding_mask,
            alive_mask[:, -1, :],
            group_indices,
        )
        q_val = ret["q_val"]
        group_mu = ret["group_mu"]
        group_log_var = ret["group_log_var"]
        group_consensus = ret["group_consensus"]
        agent_mu = ret["agent_mu"]
        agent_log_var = ret["agent_log_var"]

        actions_last = actions[:, -1].to(device=self.device, dtype=torch.int64)
        q_val = torch.gather(q_val, dim=-1, index=actions_last.unsqueeze(-1)).squeeze(-1)

        self.eval_critic.train()
        ret_critic = self.eval_critic(
            q_val,
            states,
            alive_mask,
            timestep_padding_mask[:, 0, :],
        )
        q_tot = ret_critic["q_tot"]

        # === TD Targets (Double Q-learning) ===
        # eval_agent_group selects actions, target_agent_group evaluates.
        with torch.no_grad():
            next_observations_t = torch.transpose(next_observations, 1, 2).to(
                self.device
            )
            next_states_last_np = next_states[:, -1].detach().cpu().numpy()
            next_group_indices = batch["next_group_indices"][:, -1, :].numpy()

            # -- Double Q: eval agent group selects best actions -----------
            self.eval_agent_group.eval()
            ret_next_eval = self.eval_agent_group(
                next_observations_t,
                next_states_last_np,
                next_timestep_padding_mask,
                next_alive_mask[:, -1, :],
                next_group_indices,
            )
            q_val_next_eval = ret_next_eval["q_val"]

            if use_action_mask:
                q_val_next_eval = torch.masked_fill(
                    q_val_next_eval, ~next_avail_actions, -torch.inf
                )
            best_actions = q_val_next_eval.argmax(dim=-1)

            # -- Double Q: target agent group evaluates chosen actions -----
            self.target_agent_group.reset().eval()
            ret_next_target = self.target_agent_group(
                next_observations_t,
                next_states_last_np,
                next_timestep_padding_mask,
                next_alive_mask[:, -1, :],
                next_group_indices,
            )
            q_val_next_target = ret_next_target["q_val"]
            q_val_next = q_val_next_target.gather(
                dim=-1, index=best_actions.unsqueeze(-1)
            ).squeeze(-1)

            self.target_critic.eval()
            ret_next_critic = self.target_critic(
                q_val_next,
                next_states,
                next_alive_mask,
                next_timestep_padding_mask[:, 0, :],
            )
            q_tot_next = ret_next_critic["q_tot"]

        y_tot = rewards + (1 - terminations) * self.gamma * q_tot_next
        critic_loss = torch.nn.functional.mse_loss(q_tot, y_tot.detach())

        # === SSL Reconstruction Loss ===
        if is_warmup:
            ssl_loss = torch.tensor(0.0, device=self.device)
        else:
            # Pre-generated targets provided by trainer via
            # SSLEnrichedTrajectoryDataset — no per-batch GPU↔CPU round-trip.
            targets = batch["formatted_obs"].to(
                dtype=torch.float32, device=self.device
            )
            construct_mask = batch["construct_padding_mask"].to(
                dtype=torch.bool, device=self.device
            )

            if self.recon_mode == "per_group":
                reconstruction_loss = self._recon_loss_per_group(
                    group_consensus, targets, construct_mask
                )
            else:
                reconstruction_loss = self._recon_loss_per_agent(
                    group_consensus, group_indices, targets,
                    construct_mask, alive_mask
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

            ssl_loss = reconstruction_loss + self.kl_divergence_weight * kl_divergence

        if is_warmup:
            combined_loss = critic_loss
        else:
            combined_loss = self._combine_rl_ssl_loss(critic_loss, ssl_loss)

        # === Backward Pass ===
        self.critic_optimizer.zero_grad()
        self.agent_optimizer.zero_grad()
        self.ssl_optimizer.zero_grad()

        combined_loss.backward()

        self.reduce_gradients()

        torch.nn.utils.clip_grad_norm_(self.eval_critic.parameters(), max_norm=self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.eval_agent_group.parameters(), max_norm=self.max_grad_norm)
        if not is_warmup:
            torch.nn.utils.clip_grad_norm_(self.ssl_model.parameters(), max_norm=self.max_grad_norm)

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

    def _build_recon_targets(self, observations, states, group_indices, alive_mask):
        obs_np = observations.detach().cpu().numpy()
        states_np = states.detach().cpu().numpy()
        alive_np = alive_mask.detach().cpu().numpy()

        targets_np, construct_mask_np = self.data_constructor.process(
            observations=obs_np,
            states=states_np,
            grouping=group_indices,
            alive_mask=alive_np,
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

    def _compute_ssl_loss(self, pred_set, target_set, mask=None):
        if hasattr(self.reconstruction_loss, "reconstruction_loss"):
            return self.reconstruction_loss.reconstruction_loss(
                pred_set, target_set, mask
            )
        else:
            return self.reconstruction_loss(pred_set, target_set)

    def _combine_rl_ssl_loss(self, critic_loss, ssl_loss):
        if self.loss_combination_method == "pit_loss":
            losses = torch.stack([critic_loss, ssl_loss])
            combined_loss = self.pit_loss(losses)
        else:
            combined_loss = (
                critic_loss + self.self_supervised_learning_loss_weight * ssl_loss
            )
        return combined_loss

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
