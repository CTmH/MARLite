"""Opt-in real SUMO adapter acceptance, with explicit fixture requirements."""

import os
from pathlib import Path
import pytest

if os.environ.get("MARLITE_NATIVE_SUMO") != "1":
    pytest.skip("Enable MARLITE_NATIVE_SUMO=1 for SUMO integration", allow_module_level=True)

import sumo_rl
from marlite.environment.sumo_wrapper import SUMOWrapper
from marlite.environment.padding import PadAgentView
from marlite.environment.controllers import RandomController


def test_native_heterogeneous_traffic_lights():
    fixture = Path(sumo_rl.__file__).parent / "nets" / "RESCO" / "grid4x4"
    network = Path(os.environ.get("MARLITE_SUMO_NET", str(fixture / "grid4x4.net.xml")))
    routes = Path(os.environ.get("MARLITE_SUMO_ROUTES", str(fixture / "grid4x4_1.rou.xml")))
    assert network.is_file() and routes.is_file(), "Provide MARLITE_SUMO_NET/MARLITE_SUMO_ROUTES"
    env = PadAgentView(
        SUMOWrapper(
            sumo_rl.parallel_env(
                net_file=str(network), route_file=str(routes), use_gui=False, num_seconds=60
            )
        )
    )
    try:
        observation, _ = env.reset(seed=42)
        controller = RandomController()
        controller.reset(42, env.possible_agents)
        for _ in range(3):
            spaces = {a: observation[a]["action_mask"] for a in env.agents}
            observation, _, _, _, _ = env.step(controller.actions(observation, spaces, env.agents))
        assert env.state().shape[0] == len(env.possible_agents)
    finally:
        env.close()


@pytest.mark.parametrize("recipe", ["mappo", "qplex"])
def test_sumo_real_collection_and_sequence_update(tmp_path, recipe):
    from marlite.configuration.recipe import build_experiment
    import torch

    fixture = Path(sumo_rl.__file__).parent / "nets" / "RESCO" / "grid4x4"
    config = {
        "version": 3,
        "recipe": recipe,
        "seed": 42,
        "environment": {
            "module": "sumo_rl",
            "params": {
                "net_file": os.environ.get("MARLITE_SUMO_NET", str(fixture / "grid4x4.net.xml")),
                "route_file": os.environ.get(
                    "MARLITE_SUMO_ROUTES", str(fixture / "grid4x4_1.rou.xml")
                ),
                "use_gui": False,
                "num_seconds": 60,
            },
            "wrappers": [{"type": "SUMOWrapper"}, {"type": "PadAgentView"}],
        },
        "defaults": {"hidden_dim": 8, "mixing_dim": 4, "heads": 2},
        "collector": {"max_steps": 3},
        "trainer": {"sequence_length": 2, "batch_sequences": 2},
        "run": {"iterations": 1, "train_episodes": 1, "eval_interval": None},
    }
    if recipe == "mappo":
        config["trainer"]["ppo_epochs"] = 1
    else:
        config["trainer"]["updates_per_iteration"] = 1
    experiment = build_experiment(
        config, workdir=tmp_path / recipe, device=os.environ.get("MARLITE_ACCEPTANCE_DEVICE", "cpu")
    )
    experiment.run()
    learner = experiment.trainer.learner
    assert learner.updates > 0 and all(torch.isfinite(p).all() for p in learner.actor.parameters())
