from pettingzoo.utils.env import ParallelEnv
from ..algorithm.agents.agent_group_config import AgentGroupConfig

class ParallelEnvWrapper:
    def __init__(self, **kwargs):
        self.opponent_agent_group_config = kwargs.pop('opponent_agent_group_config', None)
        self.opponent_agent_group_config = AgentGroupConfig(**self.opponent_agent_group_config)
        self.opponent_agent_group = self.opponent_agent_group_config.get_agent_group()
        self.opp_obs_queue_len = kwargs.pop('opp_obs_queue_len')
        self.env = None

    def step(self, actions):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError
    
    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()
    
    def state(self):
        return self.env.state()
    
    def observation_space(self, agent):
        return self.env.observation_space(agent)
    
    def action_space(self, agent):
        return self.env.action_space(agent)