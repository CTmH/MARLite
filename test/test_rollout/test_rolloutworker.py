import unittest
import torch
import yaml
import numpy as np
import torch.multiprocessing as mp
from copy import deepcopy

from src.algorithm.agents import QMIXAgentGroup
from src.rollout.multiprocess_rolloutworker import MultiProcessRolloutWorker
from src.algorithm.model import ModelConfig
from src.environment.env_config import EnvConfig
from src.util.optimizer_config import OptimizerConfig
from src.algorithm.agents.agent_group_config import AgentGroupConfig
import torch.nn as nn
import torch.nn.init as init
import pickle

def is_picklable(obj):
    try:
        pickle.dumps(obj)
        return True
    except (TypeError, pickle.PicklingError):
        return False
    
def init_weights(m):
    if isinstance(m, nn.Linear):
        init.constant_(m.weight, 0.5)
        if m.bias is not None:
            init.constant_(m.bias, 0)

def rollout_worker_func(worker_class,
                        worker_kwargs):
    worker = worker_class(**worker_kwargs)
    return worker.run()

class TestRolloutWorker(unittest.TestCase):

    def setUp(self):
        # Environment setup and model configuration
        self.env_config = {"module_name": "pettingzoo.mpe", "env_name": "simple_spread_v3"}
        self.env_config = EnvConfig(**self.env_config)
        # Agent group configuration
        config = """
            agent_group_config:
                type: "QMIX"
                agent_list:
                    agent_0: model1
                    agent_1: model1
                    agent_2: model1
                model_configs:
                    model1:
                        feature_extractor:
                            model_type: "Identity"
                        model:
                            model_type: "RNN"
                            input_shape: 18
                            rnn_hidden_dim: 128
                            rnn_layers: 1
                            output_shape: 5
                optimizer:
                    type: "Adam"
                    lr: 0.0005
                    weight_decay: 0.0001
        """
        config = yaml.safe_load(config)
        self.agent_group_config = AgentGroupConfig(**config['agent_group_config'])
        
        # Initialize QMIXAgents models parameters
        self.agent_group = self.agent_group_config.get_agent_group()
        for model in self.agent_group.models.values():
            model.apply(init_weights)
        for fe in self.agent_group.feature_extractors.values():
            fe.apply(init_weights)
        self.agent_model_params, self.agent_fe_params = self.agent_group.get_agent_group_params()

        self.traj_len = 5
        self.n_episodes = 2
        self.episode_limit = 10
        self.episode_queue = mp.Queue()
        self.worker = MultiProcessRolloutWorker(env=self.env_config.create_env(),
                                    agent_group = self.agent_group,
                                    n_episodes=self.n_episodes,
                                    rnn_traj_len=self.traj_len,
                                    episode_limit=self.episode_limit,
                                    epsilon=0.9,
                                    device='cpu')

    def test_rollout(self):
        # Test act method with epsilon = 0 (greedy policy)
        episode = self.worker.rollout()
        self.assertFalse(not isinstance(episode, dict))
        self.assertEqual(len(episode["rewards"]), self.episode_limit)

    def test_run(self):
        episodes = self.worker.run()
        for episode in episodes:
            self.assertFalse(not isinstance(episode, dict))
            self.assertEqual(len(episode["rewards"]), self.episode_limit)

        self.assertEqual(len(episodes), self.n_episodes)
    
    def test_pickle(self):
        self.assertTrue(is_picklable(self.agent_model_params))
        self.assertTrue(is_picklable(self.agent_fe_params))
        self.assertTrue(is_picklable(self.agent_group_config.get_agent_group()))

        config = '''
        env_config:
            module_name: "custom"
            env_name: "adversarial_pursuit_predator"
            tag_penalty: 0.0
            opp_obs_queue_len: 5
            opponent_agent_group_config:
                type: "QMIX"
                agent_list:
                    agent_0: model1
                    agent_1: model1
                    agent_2: model1
                model_configs:
                    model1:
                        feature_extractor:
                            model_type: "Identity"
                        model:
                            model_type: "RNN"
                            input_shape: 18
                            rnn_hidden_dim: 128
                            rnn_layers: 1
                            output_shape: 5
                optimizer:
                    type: "Adam"
                    lr: 0.0005
                    weight_decay: 0.0001
        '''
        config = yaml.safe_load(config)
        env_config = EnvConfig(**config['env_config'])
        self.assertTrue(is_picklable(env_config.create_env()))
        #self.assertTrue(is_picklable(self.env_config))
        #self.assertFalse(self.env_config.create_env() is self.env_config.create_env())

# Python network in agent group can not excute in multiprocessing
'''
    def test_multiprocessing(self):
        self.workers = []
        n_worker = 1
        tasks = []
        for i in range(n_worker):
            worker_kwargs = {
                'env': self.env_config.create_env(),
                'agent_group': deepcopy(self.agent_group),
                'n_episodes': self.n_episodes,
                'rnn_traj_len': self.traj_len,
                'episode_limit': self.episode_limit,
                'epsilon': 0.9,
                'device': 'cpu'
            }
            tasks.append((MultiProcessRolloutWorker, worker_kwargs))
        with mp.Pool(processes=n_worker) as pool:
            # 使用 starmap 并行执行 worker_function
            results = pool.starmap(rollout_worker_func, tasks)
'''

if __name__ == '__main__':
    unittest.main()
