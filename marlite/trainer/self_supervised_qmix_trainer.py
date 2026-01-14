import os
import torch
import datetime
import yaml
import numpy as np
from copy import deepcopy
from absl import logging
from torch.nn.modules.loss import _Loss

from marlite.algorithm.model import ModelConfig
from marlite.trainer.trainer import Trainer
from marlite.util.optimizer_config import OptimizerConfig
from marlite.util.lr_scheduler_config import LRSchedulerConfig
from marlite.util.self_supervised_data_constructor.self_supervised_data_constructor_config import SelfSupervisedDataConstructorConfig


class SelfSupervisedQMIXTrainer(Trainer):

    def __init__(self,
                 ssl_model_config: ModelConfig,
                 ssl_optimizer_config: OptimizerConfig,
                 ssl_lr_scheduler_conf: LRSchedulerConfig,
                 data_constructor_config: SelfSupervisedDataConstructorConfig,
                 reconstruction_loss: _Loss,
                 self_supervised_learning_loss_weight=1.0,
                 **kwargs):

        self.ssl_model_config = ssl_model_config
        self.data_constructor_config = data_constructor_config
        self.reconstruction_loss = reconstruction_loss
        self.self_supervised_learning_loss_weight = self_supervised_learning_loss_weight

        super().__init__(**kwargs)

        # Create data constructor from config
        self.data_constructor = self.data_constructor_config.get_data_constructor()

        self.ssl_model = ssl_model_config.get_model()
        self.best_ssl_model_params = deepcopy(self.ssl_model.state_dict())
        self._cached_ssl_model_params = deepcopy(self.ssl_model.state_dict())

        # Initialize self_supervised_model optimizer and scheduler separately
        self.ssl_optimizer_config = ssl_optimizer_config
        self.ssl_lr_scheduler_conf = ssl_lr_scheduler_conf

        self._init_ssl_optimizer()

        # Create separate learning rate schedulers for critic and self_supervised_model
        if self.ssl_lr_scheduler_conf:
            self.ssl_lr_scheduler = self.ssl_lr_scheduler_conf.get_lr_scheduler(self.ssl_optimizer)
        else:
            self.ssl_lr_scheduler = None

    def _init_ssl_optimizer(self):
         # Create separate optimizers for self-supervised learning
        ssl_params_optim = [
            {'params': self.ssl_model.parameters()},
        ]
        self.ssl_optimizer = self.ssl_optimizer_config.get_optimizer(ssl_params_optim)
        return self

    def save_current_model(self, checkpoint: str):
        """Save current model including self_supervised_model parameters"""
        agent_path = os.path.join(self.checkpointdir, checkpoint, "agent")
        os.makedirs(agent_path, exist_ok=True)
        self.eval_agent_group.to("cpu")
        self.eval_agent_group.save_params(agent_path)

        critic_path = os.path.join(self.checkpointdir, checkpoint, "critic")
        os.makedirs(critic_path, exist_ok=True)
        self.eval_critic.to("cpu")
        critic_params = self.eval_critic.state_dict()
        torch.save(critic_params, os.path.join(critic_path, "critic.pth"))

        # Save self_supervised_model parameters
        ssl_model_path = os.path.join(self.checkpointdir, checkpoint, "self_supervised_model")
        os.makedirs(ssl_model_path, exist_ok=True)
        self.ssl_model.to("cpu")
        ssl_model_params = self.ssl_model.state_dict()
        torch.save(ssl_model_params, os.path.join(ssl_model_path, "self_supervised_model.pth"))
        return self

    def load_checkpoint(self, checkpoint: str):
        """Load checkpoint including self_supervised_model parameters"""
        self.best_metrics = {key: -np.inf for key in self.eval_metric_list}
        agent_path = os.path.join(self.checkpointdir, checkpoint, "agent")
        self.eval_agent_group.to("cpu")
        self.eval_critic.to("cpu")
        self.ssl_model.to("cpu")
        self.eval_agent_group.load_params(agent_path)
        critic_path = os.path.join(self.checkpointdir, checkpoint, "critic", "critic.pth")
        self.eval_critic.load_state_dict(torch.load(critic_path, weights_only=True))

        # Load self_supervised_model parameters if self_supervised_model exists
        ssl_model_path = os.path.join(self.checkpointdir, checkpoint, "self_supervised_model", "self_supervised_model.pth")
        if os.path.exists(ssl_model_path):
            self.ssl_model.load_state_dict(torch.load(ssl_model_path, weights_only=True))

        self.best_agent_group_params = self.eval_agent_group.get_agent_group_params()
        self.best_critic_params = deepcopy(self.eval_critic.state_dict())
        self.best_ssl_model_params = deepcopy(self.ssl_model.state_dict())
        self._cached_agent_group_params = self.eval_agent_group.get_agent_group_params()
        self._cached_critic_params = deepcopy(self.eval_critic.state_dict())
        self._cached_ssl_model_params = deepcopy(self.ssl_model.state_dict())
        self.update_target_model_params()
        return self

    def save_best_model(self):
        """Save best model including self_supervised_model parameters"""
        self.eval_agent_group.set_agent_group_params(self.best_agent_group_params)
        self.eval_critic.load_state_dict(self.best_critic_params)
        self.ssl_model.load_state_dict(self.best_ssl_model_params)
        self.save_current_model(checkpoint = 'best')
        return self

    def update_target_model_params(self):
        """Update target model parameters including self_supervised_model"""
        agent_group_params = self.eval_agent_group.get_agent_group_params()
        self.target_agent_group.set_agent_group_params(agent_group_params)
        critic_params = deepcopy(self.eval_critic.state_dict())  # Update critic parameters
        self.target_critic.load_state_dict(critic_params)
        ssl_model_params = deepcopy(self.ssl_model.state_dict())
        self.ssl_model.load_state_dict(ssl_model_params)
        return self

    def train(self,
              epochs,
              target_first_metric,
              eval_interval=1,
              update_target_interval=1,
              batch_size=64,
              learning_times_per_epoch=1,
              ssl_batch_size=64,
              ssl_learning_times_per_epoch=1):
        # Training loop
        for epoch in range(epochs):
            self.current_epoch = epoch

            logging.info(f"Epoch {epoch}: Collecting experiences")
            self.collect_experience(epsilon=self.epsilon.get_value(epoch))

            if self.sample_mode == 'ratio':
                sample_ratio = self.sample_ratio.get_value(epoch)
                sample_size = len(self.replaybuffer.buffer) * sample_ratio
                sample_size = round(sample_size)
            else:  # self.sample_mode == 'direct'
                sample_size = round(self.sample_ratio.get_value(epoch))
            sample_size = min(sample_size, len(self.replaybuffer.buffer))

            # Learn and update eval model
            agent_group_lr = self.eval_agent_group.optimizer.param_groups[0]['lr']
            critic_lr = self.optimizer.param_groups[0]['lr']
            ssl_lr = self.ssl_optimizer.param_groups[0]['lr']
            logging.info(f"Epoch {epoch}: Batch size: {batch_size}, Critic learning rate: {critic_lr:.8f}, Self-supervised learning rate: {ssl_lr:.8f}, Agent learning rate: {agent_group_lr:.8f}")
            logging.info(f"Epoch {epoch}: Learning {learning_times_per_epoch} times per epoch ...")
            logging.info(f"Epoch {epoch}: Self-Supervised Learning ...")
            ssl_loss = self.self_supervised_learn(sample_size=sample_size, batch_size=ssl_batch_size, times=ssl_learning_times_per_epoch)
            logging.info(f"Epoch {epoch}: Self-Supervised Learning Loss {ssl_loss:.4f}")
            logging.info(f"Epoch {epoch}: Reinforcement Learning ...")
            loss = self.learn(sample_size=sample_size, batch_size=batch_size, times=learning_times_per_epoch)
            logging.info(f"Epoch {epoch}: Reinforcement Learning Loss {loss:.4f}")

            # Save checkpoint
            checkpoint_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"checkpoint_{checkpoint_time}_{epoch}"
            self.save_current_model(checkpoint_name)
            logging.info(f"Checkpoint saved at {checkpoint_name}")

            result = self.evaluate()
            metrics = {key: result[key]['mean'] for key in self.eval_metric_list}
            first_metric = next(iter(metrics.values()))
            first_metric_name = next(iter(metrics.keys()))
            self.save_intermediate_results(epoch, result)

            # Update learning rate schedulers
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(first_metric)
            elif isinstance(self.lr_scheduler, torch.optim.lr_scheduler.LRScheduler):
                self.lr_scheduler.step()

            if isinstance(self.ssl_lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.ssl_lr_scheduler.step(first_metric)
            elif isinstance(self.ssl_lr_scheduler, torch.optim.lr_scheduler.LRScheduler):
                self.ssl_lr_scheduler.step()

            self.eval_agent_group.lr_scheduler_step(first_metric)

            cache_params = []
            update_best = []
            for metric_name in self.eval_metric_list:
                metric = metrics[metric_name]
                best_metric = self.best_metrics[metric_name]
                #if mean_reward >= self.best_mean_reward * (1 - self.eval_threshold)
                cache_params.append((metric - best_metric) / max(abs(best_metric), 1) >= -self.eval_threshold)
                update_best.append(metric >= best_metric)
            cache_params = np.array(cache_params, dtype=np.bool)
            update_best = np.array(update_best, dtype=np.bool)

            if cache_params.any():
                self._cached_agent_group_params = self.eval_agent_group.get_agent_group_params()
                self._cached_critic_params = deepcopy(self.eval_critic.state_dict())
                self._cached_ssl_model_params = deepcopy(self.ssl_model.state_dict())
                logging.info(f"Epoch {epoch}: Cached parameters updated with current parameters.")

            if update_best.any():
                self.best_metrics = metrics
                self.best_agent_group_params = self.eval_agent_group.get_agent_group_params()
                self.best_critic_params = deepcopy(self.eval_critic.state_dict())
                self.best_ssl_model_params = deepcopy(self.ssl_model.state_dict())
                logging.info(f"Epoch {epoch}: New best {first_metric_name}: {first_metric:.4f}")

            if first_metric >= target_first_metric:
                logging.info(f"Epoch {epoch}: {first_metric_name} reached: {first_metric:.4f} >= {target_first_metric:.4f}")
                break

            if epoch % eval_interval == 0:
                self.eval_agent_group.set_agent_group_params(self._cached_agent_group_params)
                self.eval_critic.load_state_dict(self._cached_critic_params)
                self.ssl_model.load_state_dict(self._cached_ssl_model_params)
                self.update_target_model_params()
                logging.info(f"Epoch {epoch}: Eval model and Target model updated with cached parameters.")

            if epoch % update_target_interval == 0:
                self.update_target_model_params()
                logging.info(f"Epoch {epoch}: Target model updated with eval model parameters.")

        logging.info(f"Best strategy: {yaml.dump(self.best_metrics, default_flow_style=False, sort_keys=False)}")
        self.save_best_model()
        return self.best_metrics

    def self_supervised_learn(self, sample_size: float, batch_size: int, times: int):
        raise NotImplementedError