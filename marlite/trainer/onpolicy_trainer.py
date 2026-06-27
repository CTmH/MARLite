import os
import yaml
import torch
import datetime
import numpy as np
from absl import logging

from marlite.trainer.trainer import Trainer
from marlite.util.serialization import (
    serialize_to_buffer,
    deserialize_from_buffer,
    get_state_dict,
    load_state_dict_into,
)


class OnPolicyTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_multi_gpu()
        self._compile_eval_models()

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
