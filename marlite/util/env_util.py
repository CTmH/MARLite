import numpy as np
from typing import Tuple, Dict, Any
from marlite.algorithm.model import TimeSeqModel
'''
def obs_preprocess(observations: list, agent_model_dict: dict, models: dict, rnn_traj_len: int) -> Dict[str, Any]:
        agents = agent_model_dict.keys()
        processed_obs = {agent : [] for agent in agents}
        for agent, model_name in agent_model_dict.items():
            if isinstance(models[model_name], TimeSeqModel):
                obs_len = len(observations)
                if obs_len < rnn_traj_len:
                    padding_length = rnn_traj_len - obs_len
                    obs_padding = [np.zeros_like(observations[-1][agent]) for _ in range(padding_length)]
                    obs = obs_padding + [o[agent] for o in observations[-rnn_traj_len:]]
                else:
                    obs = [o[agent] for o in observations[-rnn_traj_len:]]
            else:
                obs = [observations[-1].get(agent)]
            processed_obs[agent] = np.array(obs)
        return processed_obs
'''
def obs_preprocess(observations: list, agent_model_dict: dict, models: dict, rnn_traj_len: int) -> Tuple[Dict[str, Any], np.ndarray]:
        agents = agent_model_dict.keys()
        processed_obs = {agent : [] for agent in agents}
        traj_padding_mask = {agent: [] for agent in agents}
        for agent, _ in agent_model_dict.items():
            obs_len = len(observations)
            if obs_len < rnn_traj_len:
                padding_length = rnn_traj_len - obs_len
                obs_padding = [np.zeros_like(observations[-1][agent]) for _ in range(padding_length)]
                obs = obs_padding + [o[agent] for o in observations[-rnn_traj_len:]]
                traj_padding_mask = np.array([False] * padding_length
                                + [True] * len(observations[-rnn_traj_len:]), dtype=np.bool)
            else:
                obs = [o[agent] for o in observations[-rnn_traj_len:]]
                traj_padding_mask = np.ones(rnn_traj_len, dtype=np.bool)
            processed_obs[agent] = np.array(obs)
        return processed_obs, traj_padding_mask

def ensure_all_agents_present(data_dict: dict, default_values: dict) -> Dict[str, Any]:
    """
    Ensure that the dictionary contains all possible agents.
    If any agent is missing, add it with the corresponding default value.
    Maintains the order of possible_agents.
    """

    result = {}
    for agent in default_values.keys():
        if agent in data_dict:
            result[agent] = data_dict[agent]
        elif agent in default_values:
            result[agent] = default_values[agent]
        else:
            # Fallback: use first available value as template
            if data_dict:
                first_value = next(iter(data_dict.values()))
                result[agent] = np.zeros_like(first_value)
            elif default_values:
                first_default = next(iter(default_values.values()))
                result[agent] = np.zeros_like(first_default)
    return result
