import unittest
import numpy as np
from multiprocessing.shared_memory import SharedMemory

from marlite.replaybuffer.normal_replaybuffer import NormalReplayBuffer
from marlite.util.trajectory_dataset import TrajectoryDataLoader, NUMERIC_ATTR, DYNAMIC_LEN_ATTR, OBJ_ATTR
from marlite.algorithm.agents import AgentGroupConfig
from marlite.rollout.multiprocess_rollout import multiprocess_rollout
from marlite.environment import EnvConfig
from marlite.util.serialization import serialize_to_buffer


def _create_agent_group_config(env):
    env.reset()
    agents = {env.agents[i]: ["RNN0", "RNN0", "RNN1"][i] for i in range(len(env.agents))}
    obs_shape = env.observation_space(env.agents[0]).shape[0]
    action_shape = env.action_space(env.agents[0]).n
    model_layers = {
        "model_type": "RNN",
        "input_shape": obs_shape,
        "rnn_hidden_dim": 128,
        "output_shape": action_shape,
    }
    agent_group_cfg = {
        "type": "QMIX",
        "agent_list": agents,
        "model_configs": {
            name: {
                "model": model_layers,
                "feature_extractor": {"model_type": "Identity"},
            }
            for name in ("RNN0", "RNN1")
        },
    }
    agent_group_config = AgentGroupConfig(**agent_group_cfg)
    agent_group = agent_group_config.get_agent_group()
    serialized_params = serialize_to_buffer(agent_group.state_dict())
    shm = SharedMemory(create=True, size=len(serialized_params))
    shm.buf[: len(serialized_params)] = serialized_params
    shm_info = (shm.name, len(serialized_params))
    return agent_group_config, shm, shm_info


def _setup_env():
    env_config = EnvConfig(**{"module_name": "mpe2", "env_name": "simple_spread_v3"})
    env = env_config.create_env()
    return env_config, env


class TestTrajectoryDataset(unittest.TestCase):

    def setUp(self):
        self.traj_len = 5
        self.episode_limit = 10
        self.n_episodes = 5
        self.env_config, self.env = _setup_env()
        self.agent_group_config, self.shm, self.shm_info = _create_agent_group_config(self.env)
        self.env.close()
        self.buffer = NormalReplayBuffer(capacity=10, traj_len=self.traj_len)
        episode = multiprocess_rollout(
            self.env_config, self.agent_group_config, self.shm_info,
            rnn_traj_len=self.traj_len, episode_limit=self.episode_limit,
            epsilon=0.9, device="cpu",
        )
        self.buffer.add_episode(episode)
        self.dataset = self.buffer.sample(10)

    def tearDown(self):
        self.shm.close()
        self.shm.unlink()

    def test_getitem_normal_case(self):
        for sample in self.dataset:
            self.assertEqual(len(sample['observations']), self.traj_len)
            self.assertEqual(len(sample['actions']), self.traj_len)
            self.assertEqual(len(sample['rewards']), self.traj_len)
            self.assertEqual(len(sample['states']), self.traj_len)
            self.assertEqual(len(sample['edge_indices']), self.traj_len)
            self.assertTrue(isinstance(sample['observations'][0], np.ndarray))
            self.assertTrue(isinstance(sample['actions'][0], np.ndarray))
            self.assertTrue(isinstance(sample['rewards'][0], np.ndarray))
            self.assertTrue(isinstance(sample['states'][0], np.ndarray))


class TestTrajectoryDataloader(unittest.TestCase):

    def setUp(self):
        self.traj_len = 5
        self.episode_limit = 10
        self.n_episodes = 5
        self.env_config, self.env = _setup_env()
        self.agent_group_config, self.shm, self.shm_info = _create_agent_group_config(self.env)
        self.env.close()
        self.buffer = NormalReplayBuffer(capacity=10, traj_len=self.traj_len)
        episode = multiprocess_rollout(
            self.env_config, self.agent_group_config, self.shm_info,
            rnn_traj_len=self.traj_len, episode_limit=self.episode_limit,
            epsilon=0.9, device="cpu",
        )
        self.buffer.add_episode(episode)
        self.dataset = self.buffer.sample(10)
        self.dataloader = TrajectoryDataLoader(dataset=self.dataset, batch_size=3, shuffle=True)

    def tearDown(self):
        self.shm.close()
        self.shm.unlink()

    def test_get_batch(self):
        for batch in self.dataloader:
            for key in batch.keys():
                self.assertIn(key, NUMERIC_ATTR + DYNAMIC_LEN_ATTR + OBJ_ATTR)


if __name__ == '__main__':
    unittest.main()
