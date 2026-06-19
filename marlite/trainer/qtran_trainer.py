import os
import torch
import torch.nn.functional as F
import datetime
import yaml
import numpy as np
from absl import logging
from typing import Optional
from tqdm import tqdm

from marlite.trainer.offpolicy_trainer import OffPolicyTrainer
from marlite.algorithm.critic.state_value_config import StateValueConfig
from marlite.util.optimizer_config import OptimizerConfig
from marlite.util.lr_scheduler_config import LRSchedulerConfig
from marlite.util.trajectory_dataset import TrajectoryDataLoader
from marlite.util.serialization import (
    serialize_to_buffer,
    deserialize_from_buffer,
    get_state_dict,
    load_state_dict_into,
)


class QTRANTrainer(OffPolicyTrainer):
    def __init__(
        self,
        v_net_config: StateValueConfig,
        v_optimizer_config: OptimizerConfig,
        v_lr_scheduler_conf: Optional[LRSchedulerConfig],
        lambda_opt: float = 1.0,
        lambda_nopt: float = 1.0,
        **kwargs,
    ):
        self.lambda_opt = lambda_opt
        self.lambda_nopt = lambda_nopt
        self.v_net_config = v_net_config
        self.v_optimizer_config = v_optimizer_config
        self.v_lr_scheduler_conf = v_lr_scheduler_conf

        self.eval_v_net = v_net_config.get_v_net()

        super().__init__(**kwargs)

        self.v_optimizer = v_optimizer_config.get_optimizer(
            self.eval_v_net.parameters()
        )

        self.v_lr_scheduler = (
            v_lr_scheduler_conf.get_lr_scheduler(self.v_optimizer)
            if v_lr_scheduler_conf
            else None
        )

        self.best_v_net_params = serialize_to_buffer(get_state_dict(self.eval_v_net))
        self._cached_v_net_params = serialize_to_buffer(get_state_dict(self.eval_v_net))

    def _create_worker_group(self):
        return None

    def learn(self, sample_size, batch_size: int, times: int = 1):
        return self._learn_single_gpu(sample_size, batch_size, times)

    def _learn_single_gpu(self, sample_size, batch_size: int, times: int = 1):
        total_loss = 0.0
        total_batches = 0

        self.eval_agent_group.to(self.train_device)
        self.eval_critic.to(self.train_device)
        self.target_agent_group.to(self.train_device)
        self.target_critic.to(self.train_device)
        self.eval_v_net.to(self.train_device)

        for _ in range(times):
            with tqdm(
                total=sample_size, desc=f"Times {1}/{times}", unit="batch"
            ) as pbar:
                dataset = self.replaybuffer.sample(sample_size)
                dataloader = TrajectoryDataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=self.n_workers,
                )
                for batch in dataloader:
                    alive_mask = batch["alive_mask"].to(dtype=torch.bool)
                    observations = batch["observations"].to(dtype=torch.float32)
                    timestep_padding_mask = batch["timestep_padding_mask"].to(
                        dtype=torch.bool
                    )
                    states = batch["states"].to(dtype=torch.float32)
                    actions = batch["actions"].to(dtype=torch.int)
                    rewards = batch["rewards"].to(dtype=torch.float32)
                    next_observations = batch["next_observations"].to(
                        dtype=torch.float32
                    )
                    next_timestep_padding_mask = batch["next_timestep_padding_mask"].to(
                        dtype=torch.bool
                    )
                    next_avail_actions = batch["next_avail_actions"]
                    next_alive_mask = batch["next_alive_mask"].to(dtype=torch.bool)
                    terminations = batch["terminations"].to(dtype=torch.bool)
                    n_agents = rewards.shape[2]

                    next_alive_mask = next_alive_mask.to(self.train_device)
                    alive_mask = alive_mask.to(self.train_device)

                    if isinstance(next_avail_actions, torch.Tensor):
                        use_action_mask = True
                        next_avail_actions = next_avail_actions[:, -1, :, :]
                        next_avail_actions = next_avail_actions.to(
                            dtype=torch.bool, device=self.train_device
                        )
                    else:
                        use_action_mask = False

                    r_last = self._aggregate_rewards(rewards[:, -1]).to(self.train_device)
                    termination_last = terminations[:, -1].prod(dim=1).to(self.train_device)

                    timestep_padding_mask = torch.stack(
                        [timestep_padding_mask] * n_agents, dim=1
                    ).to(self.train_device)
                    next_timestep_padding_mask = torch.stack(
                        [next_timestep_padding_mask] * n_agents, dim=1
                    ).to(self.train_device)

                    self.eval_agent_group.train()
                    observations = torch.transpose(observations, 1, 2).to(
                        self.train_device
                    )
                    ret = self.eval_agent_group(
                        observations, timestep_padding_mask, alive_mask[:, -1, :]
                    )
                    q_val = ret["q_val"]
                    enc_out = ret["enc_out"]
                    actions_last = actions[:, -1].to(
                        device=self.train_device, dtype=torch.int64
                    )

                    self.eval_critic.train()
                    cret = self.eval_critic(enc_out, actions_last)
                    Q_jt_per_action = cret["q_per_action"]

                    self.eval_v_net.train()
                    vret = self.eval_v_net(
                        states.to(self.train_device),
                        alive_mask,
                        timestep_padding_mask[:, 0, :],
                    )
                    v_jt = vret["v"]

                    q_jt_at_a = Q_jt_per_action.gather(
                        -1, actions_last.unsqueeze(-1)
                    ).squeeze(-1)
                    q_jt_scalar = q_jt_at_a.mean(dim=1)

                    with torch.no_grad():
                        self.eval_agent_group.eval()
                        next_observations_t = torch.transpose(
                            next_observations, 1, 2
                        ).to(self.train_device)
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
                        best_actions = q_val_next_eval.argmax(dim=-1)

                        self.target_agent_group.eval()
                        ret_next_target = self.target_agent_group(
                            next_observations_t,
                            next_timestep_padding_mask,
                            next_alive_mask[:, -1, :],
                        )
                        enc_out_next = ret_next_target["enc_out"]

                        self.target_critic.eval()
                        Q_jt_next = self.target_critic(enc_out_next, best_actions)[
                            "q_per_action"
                        ]
                        q_jt_next_at_best = (
                            Q_jt_next.gather(-1, best_actions.unsqueeze(-1))
                            .squeeze(-1)
                            .mean(dim=1)
                        )

                    y = r_last + (1 - termination_last) * self.gamma * q_jt_next_at_best
                    td_loss = F.mse_loss(q_jt_scalar, y.detach())

                    qmax_idx = q_val.argmax(dim=-1)
                    qmax = q_val.max(dim=-1).values
                    q_jt_at_qmax = Q_jt_per_action.gather(
                        -1, qmax_idx.unsqueeze(-1)
                    ).squeeze(-1)
                    is_optimal = (actions_last == qmax_idx).all(dim=1).float()
                    diff_opt = qmax.sum(1) - q_jt_at_qmax.detach().sum(1) + v_jt.squeeze(-1)
                    L_opt = (is_optimal * diff_opt.square()).mean()

                    q_actual_i = q_val.gather(
                        -1, actions_last.unsqueeze(-1)
                    ).squeeze(-1)
                    counter_sum = (q_actual_i.sum(1, keepdim=True) - q_actual_i).unsqueeze(-1)
                    Q_prime_cf = q_val + counter_sum
                    D = Q_prime_cf - Q_jt_per_action.detach() + v_jt.unsqueeze(-1)
                    D_min = D.min(dim=-1).values
                    L_nopt = (is_optimal.unsqueeze(-1) * D_min.square()).mean()

                    total_loss_batch = (
                        td_loss
                        + self.lambda_opt * L_opt
                        + self.lambda_nopt * L_nopt
                    )

                    self.agent_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    self.v_optimizer.zero_grad()
                    total_loss_batch.backward()

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

                    total_loss += total_loss_batch.detach().cpu().item()
                    total_batches += 1
                    pbar.update(batch["states"].shape[0])

        self.eval_agent_group.to("cpu")
        self.eval_critic.to("cpu")
        self.target_agent_group.to("cpu")
        self.target_critic.to("cpu")
        self.eval_v_net.to("cpu")

        torch.cuda.empty_cache()

        return total_loss / max(total_batches, 1)

    def update_target_model_params(self):
        if self.target_update_mode == "hard":
            load_state_dict_into(self.target_agent_group, get_state_dict(self.eval_agent_group))
            load_state_dict_into(self.target_critic, get_state_dict(self.eval_critic))
        else:
            self._ema_update(
                self.target_agent_group, self.eval_agent_group, self.target_update_tau
            )
            self._ema_update(self.target_critic, self.eval_critic, self.target_update_tau)
        return self

    def save_current_model(self, checkpoint: str):
        agent_path = os.path.join(self.checkpointdir, checkpoint, "agent")
        os.makedirs(agent_path, exist_ok=True)
        self.eval_agent_group.to("cpu")
        agent_params = get_state_dict(self.eval_agent_group)
        torch.save(agent_params, os.path.join(agent_path, "agent.pth"))

        critic_path = os.path.join(self.checkpointdir, checkpoint, "critic")
        os.makedirs(critic_path, exist_ok=True)
        self.eval_critic.to("cpu")
        critic_params = get_state_dict(self.eval_critic)
        torch.save(critic_params, os.path.join(critic_path, "critic.pth"))

        v_net_path = os.path.join(self.checkpointdir, checkpoint, "v_net")
        os.makedirs(v_net_path, exist_ok=True)
        self.eval_v_net.to("cpu")
        v_net_params = get_state_dict(self.eval_v_net)
        torch.save(v_net_params, os.path.join(v_net_path, "v_net.pth"))
        return self

    def load_checkpoint(self, checkpoint: str):
        self.best_metrics = {key: -np.inf for key in self.eval_metric_list}
        agent_path = os.path.join(
            self.checkpointdir, checkpoint, "agent", "agent.pth"
        )
        self.eval_agent_group.to("cpu")
        self.eval_critic.to("cpu")
        self.eval_v_net.to("cpu")
        load_state_dict_into(
            self.eval_agent_group, torch.load(agent_path, weights_only=True)
        )
        critic_path = os.path.join(
            self.checkpointdir, checkpoint, "critic", "critic.pth"
        )
        load_state_dict_into(
            self.eval_critic, torch.load(critic_path, weights_only=True)
        )
        v_net_path = os.path.join(
            self.checkpointdir, checkpoint, "v_net", "v_net.pth"
        )
        if os.path.exists(v_net_path):
            load_state_dict_into(
                self.eval_v_net, torch.load(v_net_path, weights_only=True)
            )

        self.best_agent_group_params = serialize_to_buffer(
            get_state_dict(self.eval_agent_group)
        )
        self.best_critic_params = serialize_to_buffer(get_state_dict(self.eval_critic))
        self.best_v_net_params = serialize_to_buffer(get_state_dict(self.eval_v_net))
        self._cached_agent_group_params = serialize_to_buffer(
            get_state_dict(self.eval_agent_group)
        )
        self._cached_critic_params = serialize_to_buffer(get_state_dict(self.eval_critic))
        self._cached_v_net_params = serialize_to_buffer(get_state_dict(self.eval_v_net))
        load_state_dict_into(self.target_agent_group, get_state_dict(self.eval_agent_group))
        load_state_dict_into(self.target_critic, get_state_dict(self.eval_critic))
        return self

    def save_best_model(self):
        best_dir = os.path.join(self.checkpointdir, "best")
        os.makedirs(best_dir, exist_ok=True)
        torch.save(
            deserialize_from_buffer(self.best_agent_group_params),
            os.path.join(best_dir, "agent.pth"),
        )
        torch.save(
            deserialize_from_buffer(self.best_critic_params),
            os.path.join(best_dir, "critic.pth"),
        )
        torch.save(
            deserialize_from_buffer(self.best_v_net_params),
            os.path.join(best_dir, "v_net.pth"),
        )
        return self

    def train(
        self,
        epochs,
        target_first_metric,
        rollback_interval=1,
        update_target_interval=1,
        batch_size=64,
        learning_times_per_epoch=1,
    ):
        for epoch in range(epochs):
            self.current_epoch = epoch

            logging.info(f"Epoch {epoch}: Collecting experiences")
            self.collect_experience(epsilon=self.epsilon.get_value(epoch))

            if self.sample_mode == "ratio":
                sample_ratio = self.sample_ratio.get_value(epoch)
                sample_size = len(self.replaybuffer.buffer) * sample_ratio
                sample_size = round(sample_size)
            else:
                sample_size = round(self.sample_ratio.get_value(epoch))
            sample_size = min(sample_size, len(self.replaybuffer.buffer))

            agent_group_lr = self.agent_optimizer.param_groups[0]["lr"]
            critic_lr = self.critic_optimizer.param_groups[0]["lr"]
            v_lr = self.v_optimizer.param_groups[0]["lr"]
            logging.info(
                f"Epoch {epoch}: BS={batch_size} LR=critic:{critic_lr:.6f} agent:{agent_group_lr:.6f} v:{v_lr:.6f}"
            )
            logging.info(
                f"Epoch {epoch}: Learning {learning_times_per_epoch} times per epoch ..."
            )

            loss = self.learn(
                sample_size=sample_size,
                batch_size=batch_size,
                times=learning_times_per_epoch,
            )
            logging.info(f"Epoch {epoch}: Loss {loss:.4f}")

            if self.target_update_mode == "ema":
                self.update_target_model_params()

            checkpoint_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"checkpoint_{checkpoint_time}_{epoch}"
            self.save_current_model(checkpoint_name)
            logging.info(f"Checkpoint saved at {checkpoint_name}")

            result = self.evaluate()
            metrics = {key: result[key]["mean"] for key in self.eval_metric_list}
            first_metric = next(iter(metrics.values()))
            first_metric_name = next(iter(metrics.keys()))
            self.save_intermediate_results(epoch, result)

            if isinstance(
                self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.lr_scheduler.step(first_metric)
            elif isinstance(self.lr_scheduler, torch.optim.lr_scheduler.LRScheduler):
                self.lr_scheduler.step()

            if isinstance(
                self.agent_lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.agent_lr_scheduler.step(first_metric)
            elif isinstance(
                self.agent_lr_scheduler, torch.optim.lr_scheduler.LRScheduler
            ):
                self.agent_lr_scheduler.step()

            if isinstance(
                self.v_lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.v_lr_scheduler.step(first_metric)
            elif isinstance(
                self.v_lr_scheduler, torch.optim.lr_scheduler.LRScheduler
            ):
                self.v_lr_scheduler.step()

            cache_params = []
            update_best = []
            for metric_name in self.eval_metric_list:
                metric = metrics[metric_name]
                best_metric = self.best_metrics[metric_name]
                cache_params.append(
                    (metric - best_metric) / max(abs(best_metric), 1)
                    >= -self.update_cache_threshold
                )
                update_best.append(metric >= best_metric)
            cache_params = np.array(cache_params, dtype=np.bool_)
            update_best = np.array(update_best, dtype=np.bool_)

            if cache_params.any():
                self._cached_agent_group_params = serialize_to_buffer(
                    get_state_dict(self.eval_agent_group)
                )
                self._cached_critic_params = serialize_to_buffer(
                    get_state_dict(self.eval_critic)
                )
                self._cached_v_net_params = serialize_to_buffer(
                    get_state_dict(self.eval_v_net)
                )
                logging.info(
                    f"Epoch {epoch}: Cached parameters updated with current parameters."
                )

            if update_best.any():
                self.best_metrics = metrics
                self.best_agent_group_params = serialize_to_buffer(
                    get_state_dict(self.eval_agent_group)
                )
                self.best_critic_params = serialize_to_buffer(get_state_dict(self.eval_critic))
                self.best_v_net_params = serialize_to_buffer(get_state_dict(self.eval_v_net))
                logging.info(
                    f"Epoch {epoch}: New best {first_metric_name}: {first_metric:.4f}"
                )

            if first_metric >= target_first_metric:
                logging.info(
                    f"Epoch {epoch}: {first_metric_name} reached: {first_metric:.4f} >= {target_first_metric:.4f}"
                )
                break

            if epoch % rollback_interval == 0:
                load_state_dict_into(
                    self.eval_agent_group,
                    deserialize_from_buffer(self._cached_agent_group_params),
                )
                load_state_dict_into(
                    self.eval_critic,
                    deserialize_from_buffer(self._cached_critic_params),
                )
                load_state_dict_into(
                    self.eval_v_net,
                    deserialize_from_buffer(self._cached_v_net_params),
                )
                load_state_dict_into(self.target_agent_group, get_state_dict(self.eval_agent_group))
                load_state_dict_into(self.target_critic, get_state_dict(self.eval_critic))
                logging.info(
                    f"Epoch {epoch}: Rolled back eval+v_net+target to cached parameters."
                )

            if epoch % update_target_interval == 0 and self.target_update_mode != "ema":
                self.update_target_model_params()
                logging.info(
                    f"Epoch {epoch}: Target model updated via '{self.target_update_mode}' mode."
                )

        logging.info(
            f"Best strategy: {yaml.dump(self.best_metrics, default_flow_style=False, sort_keys=False)}"
        )
        self.save_best_model()
        return self.best_metrics
