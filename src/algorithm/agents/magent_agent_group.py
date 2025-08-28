import numpy as np
from typing import Dict, Any
from .agent_group import AgentGroup

class MagentPreyAgentGroup(AgentGroup):
    '''
    agents: Dict[(agent_name(str), model_name(str))]
    model_configs: Dict[model_name(str), ModelConfig]
    '''
    def __init__(self, agents: Dict[str, str]) -> None:
        self.agents = agents

    def forward(self, observations: Dict[str, np.ndarray], eval_mode=True) -> Dict[str, Any]:
        return {'q_val': None}

    def act(self, observations: Dict[str, np.ndarray], avail_actions: Dict, epsilon: int) -> Dict[str, Any]:
        obstacle_and_other_team_presence = {key: value[-1,:,:,0] + value[-1,:,:,3] for key, value in observations.items()} # value: (T*obs_len*obs_len*F)
    
        # (num_agents, n, n)
        o_tensor = np.stack(list(obstacle_and_other_team_presence.values()))
        batch_size, n, m = o_tensor.shape
        # Centre Coord
        cx = n // 2
        cy = m // 2
        
        offsets = np.array([(-1,-1), (-1,0), (-1,1),
                        (0,-1),  (0,0),  (0,1),
                        (1,-1),  (1,0),  (1,1)])
        
        # Get all agent coordinates (batch_size, m, 2)
        points = [np.argwhere(o > 0) for o in o_tensor]
        
        # Vectorized calculation of Manhattan distance sum
        grid_coords = offsets + (cx, cy)
        dist_sums = np.zeros((batch_size, 9))
        
        for i in range(batch_size):
            if len(points[i]) > 0:
                diff = np.abs(grid_coords[:, None] - points[i][None]) # grid_coords shape changes to (9, 1, 2), points shape changes to (1, N, 2)
                # Calculate the distance from each grid point to all targets
                dist_sums[i] = np.sum(np.sum(diff, axis=-1), axis=-1)
        
        # Find the index of the position with the maximum distance sum
        max_indices = np.argmax(dist_sums, axis=1)
        
        # Construct action dictionary
        actions = {agent: max_indices[i] 
                for i, agent in enumerate(obstacle_and_other_team_presence.keys())}
    
        return {"actions": actions}