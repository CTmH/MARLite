import gymnasium as gym
import rware
from ray.rllib.env.multi_agent_env import make_multi_agent

ma_cartpole_cls = make_multi_agent("rware-medium-6ag-hard-v2")
ma_cartpole = ma_cartpole_cls({"num_agents": 2})

obs = ma_cartpole.reset()
print(obs[0])
