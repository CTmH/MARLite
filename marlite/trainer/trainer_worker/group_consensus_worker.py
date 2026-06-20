import io
import torch
import torch.distributed as dist
from typing import Any, Dict

from marlite.algorithm.agents import AgentGroupConfig
from marlite.algorithm.critic import CriticConfig
from marlite.util.optimizer_config import OptimizerConfig
from marlite.trainer.trainer_worker.offpolicy_worker import OffPolicyWorker


class GroupConsensusWorker(OffPolicyWorker):
    critic_optimizer: torch.optim.Optimizer
    agent_optimizer: torch.optim.Optimizer

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
        kl_divergence_weight: float = 0.005,
        warmup_epochs: int = 0,
        consensus_mode: str = "vae",
        **kwargs,
    ):
        super().__init__(worker_id, device_id, rank, world_size, init_method)
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.kl_divergence_weight = kl_divergence_weight
        self.warmup_epochs = warmup_epochs
        self.consensus_mode = consensus_mode

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

    def move_to_device(self, device: str):
        if self.eval_agent_group is not None:
            self.eval_agent_group.to(device)
        if self.target_agent_group is not None:
            self.target_agent_group.to(device)
        if self.eval_critic is not None:
            self.eval_critic.to(device)
        if self.target_critic is not None:
            self.target_critic.to(device)
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

    def train_step(self, batch: Dict[str, Any]) -> float:
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
        epoch = batch.get("epoch", 0)
        is_warmup = epoch < self.warmup_epochs

        next_alive_mask = next_alive_mask.to(self.device)
        alive_mask = alive_mask.to(self.device)

        if isinstance(next_avail_actions, torch.Tensor):
            use_action_mask = True
            next_avail_actions = next_avail_actions[:, -1, :, :]
            next_avail_actions = next_avail_actions.to(
                dtype=torch.bool, device=self.device
            )
        else:
            use_action_mask = False

        r_last = self._aggregate_rewards(rewards[:, -1]).to(self.device)
        termination_last = terminations[:, -1].prod(dim=-1).to(self.device)

        timestep_padding_mask = torch.stack(
            [timestep_padding_mask] * n_agents, dim=1
        ).to(self.device)
        next_timestep_padding_mask = torch.stack(
            [next_timestep_padding_mask] * n_agents, dim=1
        ).to(self.device)

        self.eval_agent_group.reset().train()
        observations_t = torch.transpose(observations, 1, 2).to(self.device)
        states_t = states.to(self.device)

        states_last = states_t[:, -1]

        ret = self.eval_agent_group(
            observations_t,
            states_last,
            timestep_padding_mask,
            alive_mask[:, -1, :],
        )
        q_val = ret["q_val"]
        group_mu = ret["group_mu"]
        group_log_var = ret["group_log_var"]
        agent_mu = ret["agent_mu"]
        agent_log_var = ret["agent_log_var"]
        group_indices = ret["group_indices"]

        actions_last = actions[:, -1].to(device=self.device, dtype=torch.int64)
        q_val = torch.gather(q_val, dim=-1, index=actions_last.unsqueeze(-1)).squeeze(-1)

        self.eval_critic.train()
        ret_critic = self.eval_critic(
            q_val,
            states_t,
            alive_mask,
            timestep_padding_mask[:, 0, :],
            group_mu=group_mu,
            group_log_var=group_log_var,
            group_indices=group_indices,
        )
        q_tot = ret_critic["q_tot"]

        with torch.no_grad():
            # Double Q: eval agent group selects best actions
            self.eval_agent_group.reset().eval()
            next_observations_t = torch.transpose(next_observations, 1, 2).to(
                self.device
            )
            next_states_t = next_states.to(self.device)
            next_states_last = next_states_t[:, -1]

            ret_next_eval = self.eval_agent_group(
                next_observations_t,
                next_states_last,
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
            self.target_agent_group.reset().eval()
            ret_next_target = self.target_agent_group(
                next_observations_t,
                next_states_last,
                next_timestep_padding_mask,
                next_alive_mask[:, -1, :],
            )
            q_val_next = ret_next_target["q_val"].gather(
                dim=-1, index=best_actions.unsqueeze(-1)
            ).squeeze(-1)
            group_mu_next = ret_next_target.get("group_mu")
            group_log_var_next = ret_next_target.get("group_log_var")
            group_indices_next = ret_next_target.get("group_indices")

            self.target_critic.eval()
            ret_next_critic = self.target_critic(
                q_val_next,
                next_states_t,
                next_alive_mask,
                next_timestep_padding_mask[:, 0, :],
                group_mu=group_mu_next,
                group_log_var=group_log_var_next,
                group_indices=group_indices_next,
            )
            q_tot_next = ret_next_critic["q_tot"]

        y_tot = r_last + (1 - termination_last) * self.gamma * q_tot_next
        td_error = torch.nn.functional.mse_loss(q_tot, y_tot.detach())

        if is_warmup:
            kl_divergence = torch.tensor(0.0, device=self.device)
        elif self.consensus_mode == "ae":
            kl_divergence = torch.tensor(0.0, device=self.device)
        else:
            mask = alive_mask[:, -1, :].unsqueeze(-1).expand_as(agent_mu)
            kl_per_dim = 1 + agent_log_var - agent_mu.pow(2) - torch.exp(agent_log_var)
            kl_divergence = -0.5 * (kl_per_dim * mask).sum() / mask.sum().clamp(min=1)

        critic_loss = td_error + self.kl_divergence_weight * kl_divergence

        self.critic_optimizer.zero_grad()
        self.agent_optimizer.zero_grad()

        critic_loss.backward()

        self.reduce_gradients()

        torch.nn.utils.clip_grad_norm_(self.eval_critic.parameters(), max_norm=self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.eval_agent_group.parameters(), max_norm=self.max_grad_norm)

        self.critic_optimizer.step()
        self.agent_optimizer.step()

        # Per-batch target update (hard / ema / polyak)
        self._update_target_after_batch()

        return critic_loss.detach().cpu().item()

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
            loss = self.train_step(batch)
            del batch
            loss_queue.put(loss)

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
            if ack_queue:
                ack_queue.put("ACK")

        else:
            print(f"Worker {self.worker_id}: Unknown command: {repr(cmd)}", flush=True)

        return True

    def cleanup(self):
        if dist.is_initialized():
            dist.destroy_process_group()
