"""Exercise all SSL trainer/worker families with each configured mixer."""

import pytest

from test.test_trainer import test_multistep_training as integration


@pytest.mark.parametrize("mixer", ["weighted_sum", "pit_loss", "ema_grad_norm", "pcgrad"])
@pytest.mark.parametrize("family", [
    ("vaegc_mappo", "TestVAEGCMAPPOTrainer", "ssl_gc_mappo", "SSLGroupConsensusMAPPOWorker"),
    ("ae_gc_mappo", "TestAEGCMAPPOTrainer", "ssl_gc_mappo", "SSLGroupConsensusMAPPOWorker"),
    ("vae_group_consensus", "TestGroupConsensusTrainer", "ssl_group_consensus", "SSLGroupConsensusWorker"),
    ("ae_group_consensus", "TestAEGroupConsensusTrainer", "ssl_group_consensus", "SSLGroupConsensusWorker"),
    ("self_supervised_gnn", "TestVAEGraphQMIXBattle", "vae_graph", "VAEGraphQMIXWorker"),
])
def test_ssl_family_loss_mixer(family, mixer):
    integration.test_family_trainer_and_worker(*family, loss_mixer_type=mixer)
