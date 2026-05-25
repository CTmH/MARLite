import os
import yaml
import torch
import datetime
import numpy as np
from absl import logging
from typing import List, Union, Optional

from marlite.trainer.trainer import Trainer
from marlite.environment import EnvConfig
from marlite.rollout import RolloutManagerConfig
from marlite.replaybuffer import ReplayBufferConfig
from marlite.algorithm.agents import AgentGroupConfig
from marlite.algorithm.critic import CriticConfig
from marlite.util.optimizer_config import OptimizerConfig
from marlite.util.lr_scheduler_config import LRSchedulerConfig
from marlite.util.scheduler import Scheduler
from marlite.algorithm.critic.mixer import Mixer as MixerCritic
from marlite.analyzer import AnalyzerConfig
from marlite.util.serialization import (
    serialize_to_buffer,
    deserialize_from_buffer,
    get_state_dict,
    load_state_dict_into,
)


class OffPolicyTrainer(Trainer):
    def __init__(
        self,
        epsilon_scheduler: Scheduler = None,
        eval_epsilon: float = 0.01,
        **kwargs,
    ):
        self.epsilon = epsilon_scheduler
        self.eval_epsilon = eval_epsilon
        super().__init__(**kwargs)
        if not isinstance(self.eval_critic, MixerCritic):
            raise TypeError(
                "Mixer subclass required"
            )

        self.target_agent_group = self.agent_group_config.get_agent_group()
        self.best_agent_group_params = serialize_to_buffer(
            get_state_dict(self.eval_agent_group)
        )
        load_state_dict_into(
            self.target_agent_group,
            deserialize_from_buffer(self.best_agent_group_params),
        )
        self._cached_agent_group_params = serialize_to_buffer(
            get_state_dict(self.eval_agent_group)
        )

        self.target_critic = self.critic_config.get_critic()
        load_state_dict_into(self.target_critic, get_state_dict(self.eval_critic))
        self.best_critic_params = serialize_to_buffer(
            get_state_dict(self.eval_critic)
        )
        self._cached_critic_params = serialize_to_buffer(
            get_state_dict(self.eval_critic)
        )

        self._setup_multi_gpu()

        if self.compile_models and not self.use_multi_gpu:
            logging.info("Compiling models...")
            self.target_agent_group = torch.compile(
                self.target_agent_group.to(self.train_device)
            ).to("cpu")
            self.target_critic = torch.compile(
                self.target_critic.to(self.train_device)
            ).to("cpu")

    def _add_target_params_for_sync(self, trainable_params):
        trainable_params["target_agent_group"] = get_state_dict(self.target_agent_group)
        trainable_params["target_critic"] = get_state_dict(self.target_critic)

    def _sync_eval_params_from_workers(self):
        if self.worker_group is None:
            return
        eval_params = self.worker_group.read_params_from_worker0()
        load_state_dict_into(
            self.eval_agent_group, eval_params["eval_agent_group"]
        )
        load_state_dict_into(self.eval_critic, eval_params["eval_critic"])

    def evaluate(self):
        return super().evaluate(eval_epsilon=self.eval_epsilon)

    def update_target_model_params(self):
        load_state_dict_into(
            self.target_agent_group,
            get_state_dict(self.eval_agent_group),
        )
        load_state_dict_into(
            self.target_critic,
            get_state_dict(self.eval_critic),
        )
        return self

    def save_best_model(self):
        """Write cached best agent and critic params directly to disk."""
        import os
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
        return self

    def load_checkpoint(self, checkpoint: str):
        super().load_checkpoint(checkpoint)
        self.best_agent_group_params = serialize_to_buffer(
            get_state_dict(self.eval_agent_group)
        )
        self.best_critic_params = serialize_to_buffer(
            get_state_dict(self.eval_critic)
        )
        self._cached_agent_group_params = serialize_to_buffer(
            get_state_dict(self.eval_agent_group)
        )
        self._cached_critic_params = serialize_to_buffer(
            get_state_dict(self.eval_critic)
        )
        self.update_target_model_params()
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
            logging.info(
                f"Epoch {epoch}: Batch size: {batch_size}, Critic learning rate: {critic_lr:.8f}, Agent learning rate: {agent_group_lr:.8f}"
            )
            logging.info(
                f"Epoch {epoch}: Learning {learning_times_per_epoch} times per epoch ..."
            )

            self._sync_params_to_workers()

            loss = self.learn(
                sample_size=sample_size,
                batch_size=batch_size,
                times=learning_times_per_epoch,
            )
            logging.info(f"Epoch {epoch}: Loss {loss:.4f}")

            self._sync_eval_params_from_workers()

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

            cache_params = []
            update_best = []
            for metric_name in self.eval_metric_list:
                metric = metrics[metric_name]
                best_metric = self.best_metrics[metric_name]
                cache_params.append(
                    (metric - best_metric) / max(abs(best_metric), 1)
                    >= -self.eval_threshold
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
                logging.info(
                    f"Epoch {epoch}: Cached parameters updated with current parameters."
                )

            if update_best.any():
                self.best_metrics = metrics
                self.best_agent_group_params = serialize_to_buffer(
                    get_state_dict(self.eval_agent_group)
                )
                self.best_critic_params = serialize_to_buffer(
                    get_state_dict(self.eval_critic)
                )
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
                self.update_target_model_params()
                logging.info(
                    f"Epoch {epoch}: Eval model and Target model updated with cached parameters."
                )

            if epoch % update_target_interval == 0:
                self.update_target_model_params()
                logging.info(
                    f"Epoch {epoch}: Target model updated with eval model parameters."
                )

        logging.info(
            f"Best strategy: {yaml.dump(self.best_metrics, default_flow_style=False, sort_keys=False)}"
        )
        self.save_best_model()
        return self.best_metrics
