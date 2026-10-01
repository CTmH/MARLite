from copy import deepcopy

import torch

from marlite.algorithm.agents import AgentGroupConfig
from marlite.util.action_distribution import masked_categorical
from test.test_trainer import test_g2anet_mappo_trainer as fixtures


def test_g2anet_policy_ratio_one_and_dead_observations_ignored():
    fixture = fixtures.TestGraphMAPPOTrainer()
    fixture.setUp()
    config = deepcopy(fixture.config["agent_group"])
    config.pop("optimizer")
    config.pop("lr_scheduler", None)
    config["agent_list"] = {f"predator_{i}": "model1" for i in range(3)}
    config["graph_builder"]["n_agents"] = 3
    group = AgentGroupConfig(**config).get_agent_group()
    observations = torch.randn(3, 3, 2, 29, 10, 10)
    padding = torch.zeros(3, 3, 2, dtype=torch.bool)
    alive = torch.tensor([[True, True, True], [True, False, True], [False, False, False]])
    args = (observations, torch.zeros(3, 1), padding, alive)
    with torch.no_grad():
        collected = group.eval()(*args)["action_logits"]
        actions = collected.argmax(-1)
        old_log_prob = masked_categorical(collected).log_prob(actions)
    learned = group.train()(*args)["action_logits"]
    ratio = (masked_categorical(learned).log_prob(actions) - old_log_prob).exp()
    torch.testing.assert_close(ratio, torch.ones_like(ratio))
    changed = observations.clone()
    changed[~alive] = 10000.
    result = group(changed, args[1], padding, alive)
    torch.testing.assert_close(result["action_logits"], learned)
    learned.square().sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in group.parameters() if p.grad is not None)
    assert result["edge_indices"][2].shape == (2, 0)
