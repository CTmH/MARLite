"""Graph MAPPO trainer — on-policy PPO with graph-based agent communication.

Extends :class:`OnPolicyTrainer` with graph-specific forward calls
(5 arguments including ``states`` and ``edge_indices``) and PPO loss
computation.  Compatible with any :class:`GraphAgentGroup` subclass
(G2ANet, GNN, etc.).  Single- and multi-GPU learning loops are
implemented directly (cf. :class:`GraphQMIXTrainer` for the off-policy
analogue).
"""

import os
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm
from absl import logging

from marlite.trainer.onpolicy_trainer import OnPolicyTrainer
from marlite.trainer.trainer_worker_group.g2anet_mappo_worker_group import (
    G2ANetMAPPOWorkerGroup,
)
from marlite.util.trajectory_dataset import TrajectoryDataLoader
from marlite.util.serialization import (
    serialize_to_buffer,
    deserialize_from_buffer,
    get_state_dict,
    load_state_dict_into,
)


class GraphMAPPOTrainer(OnPolicyTrainer):
    """On-policy MAPPO trainer with graph-based multi-agent communication.

    Generic graph-enabled MAPPO trainer.  Works with any graph-based
    agent group (:class:`G2ANetMAPPOAgentGroup`, :class:`GNNAgentGroup`,
    etc.) that implements the 5-argument ``forward()`` signature with
    ``edge_indices``.

    Parameters
    ----------
    clip_epsilon : float
        PPO clip range for the importance sampling ratio.
    gae_lambda : float
        GAE lambda controlling bias-variance tradeoff.
    entropy_coef : float
        Coefficient for the entropy bonus.
    vf_coef : float
        Coefficient for the value function loss.
    max_grad_norm : float
        Maximum gradient norm for clipping.
    **kwargs :
        Forwarded to :class:`OnPolicyTrainer.__init__`.
    """

    def __init__(
        self,
        clip_epsilon: float = 0.2,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 5.0,
        **kwargs,
    ):
        self.clip_epsilon = clip_epsilon
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        super().__init__(**kwargs)

        self.best_agent_group_params = serialize_to_buffer(
            get_state_dict(self.eval_agent_group)
        )
        self.best_critic_params = serialize_to_buffer(
            get_state_dict(self.eval_critic)
        )

    # ------------------------------------------------------------------
    # Best-model persistence
    # ------------------------------------------------------------------

    def save_best_model(self):
        """Write cached best agent and critic params directly to disk."""
        best_dir = os.path.join(self.checkpointdir, "best")
        agent_dir = os.path.join(best_dir, "agent")
        os.makedirs(agent_dir, exist_ok=True)
        torch.save(
            deserialize_from_buffer(self.best_agent_group_params),
            os.path.join(agent_dir, "agent.pth"),
        )
        critic_dir = os.path.join(best_dir, "critic")
        os.makedirs(critic_dir, exist_ok=True)
        torch.save(
            deserialize_from_buffer(self.best_critic_params),
            os.path.join(critic_dir, "critic.pth"),
        )

    # ------------------------------------------------------------------
    # Multi-GPU
    # ------------------------------------------------------------------

    def _create_worker_group(self):
        if not self.use_multi_gpu:
            return None
        return G2ANetMAPPOWorkerGroup(
            device_ids=self._get_device_ids(),
            agent_group_config=self.agent_group_config,
            critic_config=self.critic_config,
            critic_optimizer_config=self.critic_optimizer_config,
            agent_optimizer_config=self.agent_optimizer_config,
            gamma=self.gamma,
            clip_epsilon=self.clip_epsilon,
            gae_lambda=self.gae_lambda,
            entropy_coef=self.entropy_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
        )

    def _sync_eval_params_from_workers(self):
        if self.worker_group is None:
            return
        eval_params = self.worker_group.read_params_from_worker0()
        load_state_dict_into(self.eval_agent_group, eval_params["eval_agent_group"])
        load_state_dict_into(self.eval_critic, eval_params["eval_critic"])

    # ------------------------------------------------------------------
    # Learning dispatch
    # ------------------------------------------------------------------

    def learn(self, sample_size, batch_size: int, times: int = 4):
        if not self.use_multi_gpu:
            return self._learn_single_gpu(sample_size, batch_size, times)
        return self._learn_multi_gpu(sample_size, batch_size, times)

    # ------------------------------------------------------------------
    # Single-GPU PPO + G2ANet
    # ------------------------------------------------------------------

    def _learn_single_gpu(self, sample_size, batch_size: int, times: int = 4):
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_batches = 0

        self.eval_agent_group.to(self.train_device)
        self.eval_critic.to(self.train_device)

        dataset = self.replaybuffer.sample(sample_size)

        for epoch in range(times):
            dataloader = TrajectoryDataLoader(
                dataset, batch_size=batch_size, shuffle=True,
                num_workers=self.n_workers,
            )
            with tqdm(
                total=sample_size, desc=f"Times {epoch + 1}/{times}", unit="batch"
            ) as pbar:
                for batch in dataloader:
                    alive_mask = batch["alive_mask"].to(dtype=torch.bool)
                    observations = batch["observations"].to(dtype=torch.float32)
                    timestep_padding_mask = batch["timestep_padding_mask"].to(
                        dtype=torch.bool, device=self.train_device
                    )
                    states = batch["states"].to(dtype=torch.float32)
                    actions = batch["actions"].to(dtype=torch.int)
                    rewards = batch["rewards"].to(dtype=torch.float32)
                    next_states = batch["next_states"].to(dtype=torch.float32)
                    next_timestep_padding_mask = batch[
                        "next_timestep_padding_mask"
                    ].to(dtype=torch.bool, device=self.train_device)
                    next_alive_mask = batch["next_alive_mask"].to(dtype=torch.bool)
                    all_log_probs = batch["all_log_probs"].to(dtype=torch.float32)
                    terminations = batch["terminations"].to(dtype=torch.bool)

                    bs = states.shape[0]
                    n_agents = rewards.shape[2]
                    device = self.train_device

                    alive_mask_d = alive_mask.to(device)
                    next_alive_mask_d = next_alive_mask.to(device)
                    states_dev = states.to(device)
                    next_states_dev = next_states.to(device)

                    # -- edge indices --
                    edge_inds = batch.get("edge_indices", [])
                    if edge_inds:
                        last_edge_idx = [ei[-1] for ei in edge_inds]
                    else:
                        last_edge_idx = []

                    # -- critic forward --
                    self.eval_critic.train()
                    v = self.eval_critic(
                        states_dev, alive_mask_d, timestep_padding_mask
                    )["v"]
                    v_last = v[:, 0]

                    with torch.no_grad():
                        v_next = self.eval_critic(
                            next_states_dev[:, -1:, ...],
                            next_alive_mask_d[:, -1:, ...],
                            next_timestep_padding_mask[:, -1:],
                        )["v"][:, 0]

                    r_last = self._aggregate_rewards(rewards[:, -1]).to(device)
                    termination_last = terminations[:, -1].prod(dim=-1).to(
                        dtype=torch.float32, device=device
                    )
                    delta = (
                        r_last + self.gamma * v_next * (1.0 - termination_last) - v_last
                    )
                    advantages_last = delta
                    returns = delta + v_last

                    # -- agent forward (G2ANet: 5 args) --
                    timestep_padding_mask_expanded = torch.stack(
                        [timestep_padding_mask] * n_agents, dim=1
                    ).to(device)
                    observations_t = torch.transpose(observations, 1, 2).to(device)

                    self.eval_agent_group.reset().train()
                    ret_agent = self.eval_agent_group(
                        observations_t,
                        states_dev,
                        timestep_padding_mask_expanded,
                        alive_mask_d[:, -1, :],
                        last_edge_idx,
                    )
                    action_logits = ret_agent["action_logits"]

                    # -- PPO actor loss --
                    actions_last = actions[:, -1].to(
                        dtype=torch.int64, device=device
                    )
                    log_probs_old = all_log_probs[:, -1, :].to(device)

                    dist = Categorical(logits=action_logits)
                    new_log_probs = dist.log_prob(actions_last)
                    entropy = dist.entropy()

                    alive_last_flag = alive_mask_d[:, -1, :].to(
                        dtype=torch.float32, device=device
                    )
                    alive_last_count = alive_last_flag.sum()

                    ratio = torch.exp(new_log_probs - log_probs_old)
                    adv_expanded = advantages_last.unsqueeze(-1).expand(-1, n_agents)
                    surr1 = ratio * adv_expanded
                    surr2 = (
                        torch.clamp(
                            ratio, 1.0 - self.clip_epsilon,
                            1.0 + self.clip_epsilon,
                        )
                        * adv_expanded
                    )
                    actor_loss = (
                        -(torch.min(surr1, surr2) * alive_last_flag).sum()
                        / max(
                            alive_last_count,
                            torch.tensor(1.0, device=device),
                        )
                    )
                    entropy_loss = (
                        -(entropy * alive_last_flag).sum()
                        / max(
                            alive_last_count,
                            torch.tensor(1.0, device=device),
                        )
                    )
                    actor_loss = actor_loss + self.entropy_coef * entropy_loss

                    # -- critic loss --
                    critic_loss = F.mse_loss(v_last, returns.detach())

                    # -- backward --
                    self.agent_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    combined_loss = actor_loss + self.vf_coef * critic_loss
                    combined_loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        self.eval_agent_group.parameters(),
                        max_norm=self.max_grad_norm,
                    )
                    torch.nn.utils.clip_grad_norm_(
                        self.eval_critic.parameters(),
                        max_norm=self.max_grad_norm,
                    )

                    self.agent_optimizer.step()
                    self.critic_optimizer.step()

                    total_actor_loss += actor_loss.detach().cpu().item()
                    total_critic_loss += critic_loss.detach().cpu().item()
                    total_batches += 1

                    pbar.update(bs)

        self.eval_agent_group.to("cpu")
        self.eval_critic.to("cpu")
        torch.cuda.empty_cache()

        avg_actor = total_actor_loss / max(total_batches, 1)
        avg_critic = total_critic_loss / max(total_batches, 1)
        return avg_actor + avg_critic * self.vf_coef

    # ------------------------------------------------------------------
    # Multi-GPU PPO + G2ANet
    # ------------------------------------------------------------------

    def _learn_multi_gpu(self, sample_size, batch_size: int, times: int = 4):
        self.worker_group.move_models_to_gpu()
        total_combined = 0.0
        total_batches = 0

        for epoch in range(times):
            dataset = self.replaybuffer.sample(sample_size)
            dataloader = TrajectoryDataLoader(
                dataset, batch_size=batch_size, shuffle=True,
                num_workers=self.n_workers,
            )
            with tqdm(
                total=sample_size, desc=f"Times {epoch + 1}/{times}", unit="batch"
            ) as pbar:
                for batch in dataloader:
                    batch["epoch"] = self.current_epoch
                    loss = self.worker_group.train_step(batch)
                    total_combined += loss
                    total_batches += 1

                    bs = batch["states"].shape[0]
                    pbar.update(bs)

        self.worker_group.move_models_to_cpu()
        torch.cuda.empty_cache()
        return total_combined / max(total_batches, 1)

    # ------------------------------------------------------------------
    # On-policy training loop
    # ------------------------------------------------------------------

    def train(
        self,
        iterations,
        target_first_metric,
        batch_size=64,
        learning_times_per_iteration=1,
    ):
        self.eval_episodes_to_replay_ratio = 1.0
        self.evaluate()

        for iteration in range(iterations):
            self.current_epoch = iteration

            sample_size = len(self.replaybuffer.buffer)
            if self.sample_mode == "ratio":
                sample_ratio = self.sample_ratio.get_value(iteration)
                sample_size = round(sample_size * sample_ratio)
            else:
                sample_size = round(self.sample_ratio.get_value(iteration))
            sample_size = min(sample_size, len(self.replaybuffer.buffer))
            if sample_size > 0:
                agent_lr = self.agent_optimizer.param_groups[0]["lr"]
                critic_lr = self.critic_optimizer.param_groups[0]["lr"]
                logging.info(
                    f"Iteration {iteration}: Batch size: {batch_size}, "
                    f"Critic lr: {critic_lr:.8f}, Agent lr: {agent_lr:.8f}"
                )
                self._sync_params_to_workers()
                loss = self.learn(
                    sample_size=sample_size,
                    batch_size=batch_size,
                    times=learning_times_per_iteration,
                )
                self._sync_eval_params_from_workers()
                logging.info(f"Iteration {iteration}: Loss {loss:.4f}")

            self.replaybuffer = self.replaybuffer_config.create_replaybuffer()
            result = self.evaluate()
            metrics = {
                key: result[key]["mean"] for key in self.eval_metric_list
            }
            first_metric = next(iter(metrics.values()))
            first_metric_name = next(iter(metrics.keys()))
            self.save_intermediate_results(iteration, result)

            if isinstance(
                self.lr_scheduler,
                torch.optim.lr_scheduler.ReduceLROnPlateau,
            ):
                self.lr_scheduler.step(first_metric)
            elif isinstance(
                self.lr_scheduler, torch.optim.lr_scheduler.LRScheduler
            ):
                self.lr_scheduler.step()

            if isinstance(
                self.agent_lr_scheduler,
                torch.optim.lr_scheduler.ReduceLROnPlateau,
            ):
                self.agent_lr_scheduler.step(first_metric)
            elif isinstance(
                self.agent_lr_scheduler,
                torch.optim.lr_scheduler.LRScheduler,
            ):
                self.agent_lr_scheduler.step()

            if first_metric >= self.best_metrics.get(
                first_metric_name, -np.inf
            ):
                self.best_metrics = metrics
                self.best_agent_group_params = serialize_to_buffer(
                    get_state_dict(self.eval_agent_group)
                )
                self.best_critic_params = serialize_to_buffer(
                    get_state_dict(self.eval_critic)
                )

            if first_metric >= target_first_metric:
                break

        logging.info(
            f"Best strategy: {yaml.dump(self.best_metrics, default_flow_style=False, sort_keys=False)}"
        )
        self.save_best_model()
        return self.best_metrics