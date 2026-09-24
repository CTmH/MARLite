import os
import yaml
import torch
import datetime
import numpy as np
from absl import logging

from marlite.trainer.trainer import Trainer
from marlite.util.trajectory_dataset import TrajectoryDataset, TrajectoryDataLoader
from marlite.util.return_estimation import bootstrap_values, generalized_advantage_estimation
from marlite.util.serialization import (
    serialize_to_buffer,
    deserialize_from_buffer,
    get_state_dict,
    load_state_dict_into,
)


class OnPolicyTrainer(Trainer):
    def __init__(self, advantage_estimator="one_step", normalize_advantages=False, **kwargs):
        """Select ``one_step`` or ``gae``; both use fixed pre-update PPO labels.

        ``gae_lambda`` is supplied by the concrete MAPPO trainer. Advantage
        normalization applies only to policy labels, never value targets.
        """
        if advantage_estimator not in ("one_step", "gae"):
            raise ValueError("advantage_estimator must be 'one_step' or 'gae'")
        if not 0 <= self.gae_lambda <= 1:
            raise ValueError("gae_lambda must be in [0, 1]")
        self.advantage_estimator = advantage_estimator
        self.normalize_advantages = normalize_advantages
        super().__init__(**kwargs)
        self._setup_multi_gpu()
        self._compile_eval_models()

    def _prepare_ppo_dataset(self, sample_size, batch_size):
        """Freeze labels before any PPO/SSL update, also for remote workers.

        Evaluate overlapping history windows in bounded chunks. Only scalar
        values/labels are retained, not GPU computation graphs or full windows.
        """
        dataset = self.replaybuffer.sample(sample_size)
        critic = self.eval_critic
        device = self.train_device
        original_device = next(critic.parameters()).device
        training = critic.training
        critic.to(device).eval()
        attrs = [
            "states", "next_states", "alive_mask", "next_alive_mask",
            "rewards", "terminations", "truncations",
        ]
        use_gae = self.advantage_estimator == "gae"
        trace_lambda = self.gae_lambda if use_gae else 0.0
        selected_positions = {}
        for episode_id, pos in dataset.sample_id_list:
            selected_positions.setdefault(episode_id, set()).add(pos)
        try:
            with torch.no_grad():
                for episode_id, selected in selected_positions.items():
                    episode = dataset.episode_buffer[episode_id]
                    length = len(episode["rewards"])
                    # GAE needs the contiguous future suffix; one-step labels
                    # need only sampled positions, not every replayed window.
                    if use_gae:
                        positions = list(range(min(selected), length))
                    else:
                        positions = sorted(selected)
                    windows = TrajectoryDataset(
                        [(episode_id, t) for t in positions],
                        dataset.episode_buffer, dataset.traj_len, required_attrs=attrs,
                    )
                    values, next_values, rewards, bootstrap = [], [], [], []
                    for batch in TrajectoryDataLoader(windows, batch_size, shuffle=False, num_workers=0):
                        alive = batch["alive_mask"].to(device=device, dtype=torch.bool)
                        value = critic(
                            batch["states"].to(device=device, dtype=torch.float32), alive,
                            batch["timestep_padding_mask"].to(device=device, dtype=torch.bool),
                        )["v"].reshape(-1)
                        values.append(value.cpu())
                        next_values.append(bootstrap_values(critic, batch, device).cpu())
                        rewards.append(self._aggregate_rewards(batch["rewards"][:, -1].float()))
                        team_terminated = batch["terminations"][:, -1].bool().all(-1)
                        has_survivors = batch["next_alive_mask"][:, -1].bool().any(-1)
                        bootstrap.append(~team_terminated & has_survivors)
                    # A timeout allows bootstrap, but never links the GAE trace
                    # to a reset or a nonconsecutive sampled transition.
                    continuation = torch.tensor([
                        i + 1 < len(positions) and positions[i + 1] == t + 1
                        and not windows.is_episode_boundary(episode, t)
                        for i, t in enumerate(positions)
                    ])
                    advantages, targets = generalized_advantage_estimation(
                        torch.cat(rewards), torch.cat(values), torch.cat(next_values),
                        torch.cat(bootstrap), continuation, self.gamma,
                        trace_lambda,
                    )
                    for i, t in enumerate(positions):
                        dataset.training_targets[episode_id, t] = {
                            "advantages": advantages[i].item(),
                            "value_targets": targets[i].item(),
                        }
        finally:
            critic.to(original_device).train(training)
        if self.normalize_advantages:
            active_ids = [key for key in dataset.sample_id_list if any(
                dataset.episode_buffer[key[0]]["alive_mask"][key[1]].values()
            )]
            if active_ids:
                sampled_advantages = torch.tensor([
                    dataset.training_targets[key]["advantages"] for key in active_ids
                ])
                mean = sampled_advantages.mean().item()
                std = sampled_advantages.std(unbiased=False).item()
                # Duplicates contribute to sample statistics, but each stored
                # label must be normalized exactly once.
                for key in set(active_ids):
                    dataset.training_targets[key]["advantages"] = (
                        dataset.training_targets[key]["advantages"] - mean
                    ) / max(std, 1e-8)
        return dataset

    def save_current_model(self, checkpoint: str):
        """On-policy: save eval models only (no target networks)."""
        agent_path = os.path.join(self.checkpointdir, checkpoint, "agent")
        os.makedirs(agent_path, exist_ok=True)
        self.eval_agent_group.to("cpu")
        torch.save(
            get_state_dict(self.eval_agent_group),
            os.path.join(agent_path, "agent.pth"),
        )

        critic_path = os.path.join(self.checkpointdir, checkpoint, "critic")
        os.makedirs(critic_path, exist_ok=True)
        self.eval_critic.to("cpu")
        torch.save(
            get_state_dict(self.eval_critic),
            os.path.join(critic_path, "critic.pth"),
        )
        return self

    def evaluate(self):
        self.eval_agent_group.eval().to("cpu")
        serialized_params = serialize_to_buffer(
            get_state_dict(self.eval_agent_group)
        )
        manager = self.rolloutmanager_config.create_manager(
            self.agent_group_config,
            serialized_params,
            self.env_config,
            epsilon=1.0,
        )
        episodes = manager.generate_episodes()
        result = self.analyzer(episodes)

        logging.info(f"Collection results:")
        for key in result.keys():
            logging.info(
                f"{key}: Mean:{result[key]['mean']:.4f} Std:{result[key].get('std', 0):.4f}"
            )

        self.eval_agent_group.to("cpu")
        torch.cuda.empty_cache()

        for episode in episodes:
            self.replaybuffer.add_episode(episode)

        return result

    def train(self, **kwargs):
        raise NotImplementedError
