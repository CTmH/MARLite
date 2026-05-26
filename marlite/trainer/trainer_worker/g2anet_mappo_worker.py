"""
G2ANet MAPPO worker implementation for multi-GPU training.

Each worker holds copies of the eval agent group and critic models.
In ``train_step()`` the worker runs the full PPO loss computation with
G2ANet graph-based agent communication, synchronising gradients via
all_reduce across workers.
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Any, Dict

from marlite.algorithm.agents import AgentGroupConfig
from marlite.algorithm.critic import CriticConfig
from marlite.util.optimizer_config import OptimizerConfig
from marlite.trainer.trainer_worker.onpolicy_worker import OnPolicyWorker


class G2ANetMAPPOWorker(OnPolicyWorker):
    """PPO worker with G2ANet graph communication for multi-GPU training."""

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
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 5.0,
        **kwargs,
    ):
        super().__init__(worker_id, device_id, rank, world_size, init_method)
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.clip_epsilon = clip_epsilon
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef

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

    def move_to_device(self, device: str):
        if self.eval_agent_group is not None:
            self.eval_agent_group.to(device)
        if self.eval_critic is not None:
            self.eval_critic.to(device)
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

    def train_step(self, batch: Dict[str, Any]) -> float:
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

        alive_mask_d = alive_mask.to(self.device)
        next_alive_mask_d = next_alive_mask.to(self.device)
        states_dev = states.to(self.device)
        next_states_dev = next_states.to(self.device)

        # -- edge indices: use last timestep from stored trajectory --
        edge_indices = batch.get("edge_indices", [])
        if edge_indices:
            last_edge_indices = [ei[-1] for ei in edge_indices]
        else:
            last_edge_indices = []

        next_edge_indices = batch.get("next_edge_indices", [])
        if next_edge_indices:
            last_next_edge_indices = [nei[-1] for nei in next_edge_indices]
        else:
            last_next_edge_indices = []

        # -- critic forward --
        self.eval_critic.train()
        v = self.eval_critic(states_dev, alive_mask_d, timestep_padding_mask)["v"]
        v_last = v[:, 0]

        with torch.no_grad():
            v_next = self.eval_critic(
                next_states_dev[:, -1:, ...],
                next_alive_mask_d[:, -1:, ...],
                next_timestep_padding_mask[:, -1:],
            )["v"][:, 0]

        r_last = rewards.sum(dim=2)[:, -1].to(self.device)
        done_last = terminations.any(dim=2)[:, -1].to(
            dtype=torch.float32, device=self.device
        )
        delta = r_last + self.gamma * v_next * (1.0 - done_last) - v_last
        advantages_last = delta
        returns = delta + v_last

        # -- agent forward (G2ANet: 5 args) --
        timestep_padding_mask_expanded = torch.stack(
            [timestep_padding_mask] * n_agents, dim=1
        ).to(self.device)
        observations_transposed = torch.transpose(observations, 1, 2).to(self.device)

        self.eval_agent_group.reset().train()
        ret_agent = self.eval_agent_group(
            observations_transposed,
            states_dev,
            timestep_padding_mask_expanded,
            alive_mask_d[:, -1, :],
            last_edge_indices,
        )
        action_logits = ret_agent["action_logits"]

        # -- PPO actor loss --
        actions_last = actions[:, -1].to(dtype=torch.int64, device=self.device)
        log_probs_old = all_log_probs[:, -1, :].to(self.device)

        dist = Categorical(logits=action_logits)
        new_log_probs = dist.log_prob(actions_last)
        entropy = dist.entropy()

        alive_last_flag = alive_mask_d[:, -1, :].to(
            dtype=torch.float32, device=self.device
        )
        alive_last_count = alive_last_flag.sum()

        ratio = torch.exp(new_log_probs - log_probs_old)
        adv_expanded = advantages_last.unsqueeze(-1).expand(-1, n_agents)
        surr1 = ratio * adv_expanded
        surr2 = (
            torch.clamp(
                ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
            )
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

        # -- critic loss --
        critic_loss = F.mse_loss(v_last, returns.detach())

        # -- backward --
        self.agent_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        combined_loss = actor_loss + self.vf_coef * critic_loss
        combined_loss.backward()

        self.reduce_gradients()

        torch.nn.utils.clip_grad_norm_(
            self.eval_agent_group.parameters(), max_norm=self.max_grad_norm
        )
        torch.nn.utils.clip_grad_norm_(
            self.eval_critic.parameters(), max_norm=self.max_grad_norm
        )

        self.agent_optimizer.step()
        self.critic_optimizer.step()

        return combined_loss.detach().cpu().item()

    def handle_command(
        self, cmd, param_queue, data_queue, loss_queue, ack_queue=None
    ) -> bool:
        if cmd == "STOP":
            self.cleanup()
            return False
        elif cmd == "SYNC_FROM_MAIN":
            import io
            params = param_queue.get()
            if isinstance(params, bytes):
                buf = io.BytesIO(params)
                params = torch.load(buf, weights_only=True)
            if "eval_agent_group" in params and self.eval_agent_group is not None:
                self.eval_agent_group.load_state_dict(
                    {k: v.clone() for k, v in params["eval_agent_group"].items()}
                )
            if "eval_critic" in params and self.eval_critic is not None:
                self.eval_critic.load_state_dict(
                    {k: v.clone() for k, v in params["eval_critic"].items()}
                )
            del params
            ack_queue.put("ACK")
        elif cmd == "BROADCAST":
            import io
            params = param_queue.get()
            if isinstance(params, bytes):
                buf = io.BytesIO(params)
                params = torch.load(buf, weights_only=True)
            if "eval_agent_group" in params and self.eval_agent_group is not None:
                self.eval_agent_group.load_state_dict(
                    {k: v.clone() for k, v in params["eval_agent_group"].items()}
                )
            if "eval_critic" in params and self.eval_critic is not None:
                self.eval_critic.load_state_dict(
                    {k: v.clone() for k, v in params["eval_critic"].items()}
                )
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
                for pg in self.critic_optimizer.param_groups:
                    pg["lr"] = lr_data["critic_lr"]
            if "agent_lr" in lr_data:
                for pg in self.agent_optimizer.param_groups:
                    pg["lr"] = lr_data["agent_lr"]
            if ack_queue:
                ack_queue.put("ACK")
        else:
            print(f"Worker {self.worker_id}: Unknown command: {repr(cmd)}", flush=True)
        return True

    def cleanup(self):
        if dist.is_initialized():
            dist.destroy_process_group()