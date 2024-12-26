from pettingzoo.utils.env import ParallelEnv

class ParallelEnvWrapper:
    def __init__(self, **kwargs):
        self.opponent_agent_group = kwargs.pop('opponent_agent_group', None)
        self.opp_obs_queue_len = kwargs.pop('opp_obs_queue_len')
        self.env = ParallelEnv()

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