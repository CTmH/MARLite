import numpy as np
import torch
from typing import Dict
from .agent_group import AgentGroup

class MagentPreyAgentGroup(AgentGroup):
    '''
    agents: Dict[(agent_name(str), model_name(str))]
    model_configs: Dict[model_name(str), ModelConfig]
    '''
    def __init__(self, agents: Dict[str, str]) -> None:
        self.agents = agents

    def forward(self, observations: Dict[str, np.ndarray], eval_mode=True) -> torch.Tensor:
        return self

    def act(self, observations: Dict[str, np.ndarray], avail_actions: Dict, epsilon: int) -> np.ndarray:
        other_team_presence = {key: value[3] for key, value in observations.items()}
    
        # (num_agents, n, n)
        o_tensor = np.stack(list(other_team_presence.values()))
        batch_size, n, m = o_tensor.shape
        # Centre Coord
        cx = n // 2
        cy = m // 2
        
        offsets = np.array([(-1,-1), (-1,0), (-1,1),
                        (0,-1),  (0,0),  (0,1),
                        (1,-1),  (1,0),  (1,1)])
        
        # 获取所有智能体位置坐标 (batch_size, m, 2)
        points = [np.argwhere(o > 0) for o in o_tensor]
        
        # 向量化计算曼哈顿距离和
        grid_coords = offsets + (cx, cy)
        dist_sums = np.zeros((batch_size, 9))
        
        for i in range(batch_size):
            if len(points[i]) > 0:
                # 计算每个网格点到所有目标的距离和
                diff = np.abs(grid_coords[:, None] - points[i][None])
                dist_sums[i] = np.sum(np.sum(diff, axis=-1), axis=-1)
        
        # 找到最大距离和的位置索引
        max_indices = np.argmax(dist_sums, axis=1)
        
        # 构建动作字典
        actions = {agent: max_indices[i] 
                for i, agent in enumerate(other_team_presence.keys())}
    
        return actions

    def set_agent_group_params(self, model_params: Dict[str, dict], feature_extractor_params: Dict[str, dict]):
        return self
    
    def get_agent_group_params(self):
        return self
    
    def zero_grad(self):
        return self
    
    def step(self):
        return self