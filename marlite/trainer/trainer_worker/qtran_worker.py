"""
QTRAN worker implementation for multi-GPU training.

This module provides :class:`QTRANWorker`, the worker counterpart of
:class:`QTRANTrainer`.  QTRAN has a third model on top of the standard
off-policy pair (eval/target agent group + eval/target critic):
``eval_v_net`` with its own ``v_optimizer``.  V is auxiliary — there is
no target V (the original QTRAN paper trains a single V).

The worker mirrors the master GPU's train_step logic exactly so that
gradients are bit-identical (modulo float-reduction order across
workers) to the single-GPU path.
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from typing import Any, Dict, Optional

from marlite.algorithm.agents import AgentGroupConfig
from marlite.algorithm.critic import CriticConfig
from marlite.algorithm.critic.state_value_config import StateValueConfig
from marlite.util.optimizer_config import OptimizerConfig
from marlite.util.lr_scheduler_config import LRSchedulerConfig
from marlite.trainer.trainer_worker.offpolicy_worker import OffPolicyWorker


class QTRANWorker(OffPolicyWorker):
    """Worker for QTRAN algorithm multi-GPU training.

    Implements ``train_step`` that runs one batch of QTRAN training
    (TD loss + L_opt + L_nopt with optional ``is_optimal_mask_mode``)
    and ``reduce_gradients`` that all-reduces gradients for **all
    three** trainable models (eval_agent_group, eval_critic,
    eval_v_net).

    The QTRAN-specific ``v_lr`` is added to ``SYNC_LR`` payloads via
    the master-side ``_extra_sync_kwargs`` hook, and consumed by
    :meth:`handle_command` below.
    """

    critic_optimizer: torch.optim.Optimizer
    agent_optimizer: torch.optim.Optimizer
    v_optimizer: torch.optim.Optimizer

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
        v_net_config: StateValueConfig,
        v_optimizer_config: OptimizerConfig,
        v_lr_scheduler_config: Optional[LRSchedulerConfig] = None,
        gamma: float = 0.95,
        max_grad_norm: float = 5.0,
        lambda_opt: float = 1.0,
        lambda_nopt: float = 1.0,
        is_optimal_mask_mode: bool = True,
        **kwargs,
    ):
        """Initialize QTRAN worker.

        Args:
            worker_id, device_id, rank, world_size, init_method:
                Distributed setup arguments forwarded to
                :class:`OffPolicyWorker`.
            agent_group_config: Configuration for the agent group.
            critic_config: Configuration for the QTRAN mixer critic
                (``Qtransform``).
            critic_optimizer_config: Optimizer for the mixer critic.
            agent_optimizer_config: Optimizer for the agent group.
            v_net_config: Configuration for the auxiliary V net
                (``StateValue``).
            v_optimizer_config: Optimizer for V net.
            v_lr_scheduler_config: Optional LR scheduler for V net.
            gamma: Discount factor for the TD target.
            max_grad_norm: Maximum gradient norm for clipping.
            lambda_opt: Weight on the L_opt term.
            lambda_nopt: Weight on the L_nopt term.
            is_optimal_mask_mode: If True, separate L_opt / L_nopt by
                per-sample optimality; if False, use a flat mean over
                the batch.
        """
        super().__init__(worker_id, device_id, rank, world_size, init_method)
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.lambda_opt = lambda_opt
        self.lambda_nopt = lambda_nopt
        self.is_optimal_mask_mode = is_optimal_mask_mode

        # Off-policy pair (eval / target agent group + critic)
        self.eval_agent_group = agent_group_config.get_agent_group()
        self.target_agent_group = agent_group_config.get_agent_group()
        self.eval_critic = critic_config.get_critic()
        self.target_critic = critic_config.get_critic()

        # V net — auxiliary, single instance (no target V)
        self.eval_v_net = v_net_config.get_v_net()

        self.eval_agent_group.train()
        self.target_agent_group.eval()
        self.eval_critic.train()
        self.target_critic.eval()
        self.eval_v_net.train()

        self.critic_optimizer = critic_optimizer_config.get_optimizer(
            self.eval_critic.parameters()
        )
        self.agent_optimizer = agent_optimizer_config.get_optimizer(
            self.eval_agent_group.parameters()
        )
        self.v_optimizer = v_optimizer_config.get_optimizer(
            self.eval_v_net.parameters()
        )
        self.v_lr_scheduler = (
            v_lr_scheduler_config.get_lr_scheduler(self.v_optimizer)
            if v_lr_scheduler_config is not None
            else None
        )

    # ------------------------------------------------------------------
    # Sync — V_net params
    # ------------------------------------------------------------------

    def sync_params_from_main(self, params):
        params = super().sync_params_from_main(params)
        if "eval_v_net" in params and self.eval_v_net is not None:
            self.eval_v_net.load_state_dict(
                {k: v.clone() for k, v in params["eval_v_net"].items()}
            )

    def get_params_for_main(self) -> Dict[str, Any]:
        params = super().get_params_for_main()
        if self.eval_v_net is not None:
            params["eval_v_net"] = {
                k: v.clone().cpu()
                for k, v in self.eval_v_net.state_dict().items()
            }
        return params

    def move_to_device(self, device: str):
        super().move_to_device(device)
        if self.eval_v_net is not None:
            self.eval_v_net.to(device)

    # ------------------------------------------------------------------
    # Gradient sync — include V_net
    # ------------------------------------------------------------------

    def reduce_gradients(self):
        """All-reduce gradients for eval_agent_group, eval_critic, eval_v_net."""
        for net in (
            self.eval_critic,
            self.eval_agent_group,
            self.eval_v_net,
        ):
            for param in net.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= self.world_size

    # ------------------------------------------------------------------
    # Learning-rate sync — add v_lr on top of critic + agent
    # ------------------------------------------------------------------

    def handle_command(
        self, cmd, param_queue, data_queue, loss_queue, ack_queue=None
    ) -> bool:
        if cmd == "SYNC_LR":
            lr_data = param_queue.get()
            if "critic_lr" in lr_data and self.critic_optimizer is not None:
                for pg in self.critic_optimizer.param_groups:
                    pg["lr"] = lr_data["critic_lr"]
            if "agent_lr" in lr_data and self.agent_optimizer is not None:
                for pg in self.agent_optimizer.param_groups:
                    pg["lr"] = lr_data["agent_lr"]
            if "v_lr" in lr_data and self.v_optimizer is not None:
                for pg in self.v_optimizer.param_groups:
                    pg["lr"] = lr_data["v_lr"]
            if ack_queue:
                ack_queue.put("ACK")
            return True
        return super().handle_command(cmd, param_queue, data_queue, loss_queue, ack_queue)

    # ------------------------------------------------------------------
    # QTRAN train step
    # ------------------------------------------------------------------

    def train_step(self, batch: Dict[str, Any]) -> float:
        """Execute one QTRAN training step on the given batch.

        Mirrors ``QTRANTrainer._learn_single_gpu`` line-for-line, with
        the master device swapped for ``self.device`` and gradient
        reduction via :meth:`reduce_gradients` before the three
        optimizer steps.
        """
        alive_mask = batch["alive_mask"].to(dtype=torch.bool)
        observations = batch["observations"].to(dtype=torch.float32)
        timestep_padding_mask = batch["timestep_padding_mask"].to(dtype=torch.bool)
        states = batch["states"].to(dtype=torch.float32)
        actions = batch["actions"].to(dtype=torch.int)
        rewards = batch["rewards"].to(dtype=torch.float32)
        next_observations = batch["next_observations"].to(dtype=torch.float32)
        next_timestep_padding_mask = batch["next_timestep_padding_mask"].to(
            dtype=torch.bool
        )
        next_avail_actions = batch["next_avail_actions"]
        next_alive_mask = batch["next_alive_mask"].to(dtype=torch.bool)
        terminations = batch["terminations"].to(dtype=torch.bool)
        n_agents = rewards.shape[2]

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
        termination_last = terminations[:, -1].prod(dim=1).to(self.device)

        timestep_padding_mask = torch.stack(
            [timestep_padding_mask] * n_agents, dim=1
        ).to(self.device)
        next_timestep_padding_mask = torch.stack(
            [next_timestep_padding_mask] * n_agents, dim=1
        ).to(self.device)

        # Eval agent group → per-agent Q values + encoder output
        self.eval_agent_group.train()
        observations = torch.transpose(observations, 1, 2).to(self.device)
        ret = self.eval_agent_group(
            observations, timestep_padding_mask, alive_mask[:, -1, :]
        )
        q_val = ret["q_val"]
        enc_out = ret["enc_out"]
        actions_last = actions[:, -1].to(device=self.device, dtype=torch.int64)

        # Eval critic → joint Q per action
        self.eval_critic.train()
        cret = self.eval_critic(enc_out, actions_last)
        Q_jt_per_action = cret["q_per_action"]

        # Eval V net → joint V
        self.eval_v_net.train()
        vret = self.eval_v_net(
            states.to(self.device),
            alive_mask,
            timestep_padding_mask[:, 0, :],
        )
        v_jt = vret["v"]

        q_jt_at_a = Q_jt_per_action.gather(-1, actions_last.unsqueeze(-1)).squeeze(-1)
        q_jt_scalar = q_jt_at_a.mean(dim=1)

        with torch.no_grad():
            self.eval_agent_group.eval()
            next_observations_t = torch.transpose(next_observations, 1, 2).to(
                self.device
            )
            ret_next_eval = self.eval_agent_group(
                next_observations_t,
                next_timestep_padding_mask,
                next_alive_mask[:, -1, :],
            )
            q_val_next_eval = ret_next_eval["q_val"]
            if use_action_mask:
                q_val_next_eval = torch.masked_fill(
                    q_val_next_eval, ~next_avail_actions, -torch.inf
                )
            next_best_actions = q_val_next_eval.argmax(dim=-1)

            self.target_agent_group.eval()
            ret_next_target = self.target_agent_group(
                next_observations_t,
                next_timestep_padding_mask,
                next_alive_mask[:, -1, :],
            )
            enc_out_next = ret_next_target["enc_out"]

            self.target_critic.eval()
            Q_jt_next = self.target_critic(enc_out_next, next_best_actions)[
                "q_per_action"
            ]
            q_jt_next_at_best = (
                Q_jt_next.gather(-1, next_best_actions.unsqueeze(-1))
                .squeeze(-1)
                .mean(dim=1)
            )

        y = r_last + (1 - termination_last) * self.gamma * q_jt_next_at_best
        td_loss = F.mse_loss(q_jt_scalar, y.detach())

        current_best_actions = q_val.argmax(dim=-1)
        qmax = q_val.max(dim=-1).values
        q_jt_at_qmax = Q_jt_per_action.gather(
            -1, current_best_actions.unsqueeze(-1)
        ).squeeze(-1)
        is_optimal = (actions_last == current_best_actions).all(dim=1).float()
        diff_opt = qmax.sum(1) - q_jt_at_qmax.detach().sum(1) + v_jt.squeeze(-1)
        diff_opt_sq = diff_opt.square()

        q_actual_i = q_val.gather(-1, actions_last.unsqueeze(-1)).squeeze(-1)
        counter_sum = (q_actual_i.sum(1, keepdim=True) - q_actual_i).unsqueeze(-1)
        Q_prime_cf = q_val + counter_sum
        D = Q_prime_cf - Q_jt_per_action.detach() + v_jt.unsqueeze(-1)
        D_min = D.min(dim=-1).values
        D_min_sq = D_min.square()

        if self.is_optimal_mask_mode:
            is_suboptimal = 1.0 - is_optimal
            denom_opt = is_optimal.sum().clamp(min=1.0)
            denom_nopt = is_suboptimal.sum().clamp(min=1.0)
            L_opt = (is_optimal * diff_opt_sq).sum() / denom_opt
            L_nopt = (is_suboptimal.unsqueeze(-1) * D_min_sq).sum() / denom_nopt
        else:
            L_opt = diff_opt_sq.mean()
            L_nopt = D_min_sq.mean()

        total_loss_batch = (
            td_loss
            + self.lambda_opt * L_opt
            + self.lambda_nopt * L_nopt
        )

        # Backward
        self.agent_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        self.v_optimizer.zero_grad()
        total_loss_batch.backward()

        # All-reduce gradients across workers
        self.reduce_gradients()

        # Clip and step
        torch.nn.utils.clip_grad_norm_(
            self.eval_agent_group.parameters(), max_norm=self.max_grad_norm
        )
        torch.nn.utils.clip_grad_norm_(
            self.eval_critic.parameters(), max_norm=self.max_grad_norm
        )
        torch.nn.utils.clip_grad_norm_(
            self.eval_v_net.parameters(), max_norm=self.max_grad_norm
        )

        self.critic_optimizer.step()
        self.v_optimizer.step()
        self.agent_optimizer.step()

        # Per-batch target update (hard / ema / polyak) — V has no target
        self._update_target_after_batch()

        return total_loss_batch.detach().cpu().item()
