from typing import Dict, Any
from pettingzoo.utils import ParallelEnv

def check_smac_victory(env: ParallelEnv, infos: Dict[str, Any]) -> bool:
    """
    Check if the game was won using SMAC victory conditions.

    Args:
        env: The environment instance
        infos: The info dictionary from the environment

    Returns:
        bool: True if the game was won, False otherwise
    """
    # Primary Method: Check the "won" flag in info dictionary
    if infos:
        for info in infos.values():
            if isinstance(info, dict) and info.get("won", False):
                return True

    return False

def check_battle_wrapper_victory(env: ParallelEnv, infos: Dict[str, Any]) -> bool:
    if len(env.opponent_agents) <= len(env.agents) * 0.8:
        return True
    return False

def always_win(env, infos) -> bool:
    """Always return True, for testing purposes"""
    return True


def always_lose(env, infos) -> bool:
    """Always return False, for testing purposes"""
    return False