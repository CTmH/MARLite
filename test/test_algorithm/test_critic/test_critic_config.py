import unittest
import yaml
from marlite.algorithm.critic import CriticConfig, QMixer, SeqQMixer
from marlite.algorithm.critic.group_consensus_mixer import GroupConsensusMixer

class TestCriticConfig(unittest.TestCase):

    def test_get_critic(self):
        config = yaml.safe_load("""
critic:
  type: "QMixer"
  model:
    model_type: QMixModel
    state_shape: 54
    input_dim: 3
    qmix_hidden_dim: 128
  feature_extractor:
    model_type: "Identity"
  optimizer:
    type: "Adam"
    lr: 0.0005
    weight_decay: 0.0001
""")
        critic_config_dict = config['critic']
        critic_config_dict.pop('optimizer')
        if 'lr_scheduler' in critic_config_dict:
            critic_config_dict.pop('lr_scheduler')
        self.critic_config = CriticConfig(**critic_config_dict)
        self.critic = self.critic_config.get_critic()
        self.assertIsInstance(self.critic, QMixer)

    def test_get_seq_critic(self):
        config = yaml.safe_load("""
critic:
  type: "SeqQMixer"
  model:
    model_type: QMixModel
    state_shape: 64
    input_dim: 5
    qmix_hidden_dim: 64
    hypernet_layers: 2
    hyper_hidden_dim: 128
  feature_extractor:
    model_type: "ResAttMaskedStateEnc"
    input_dim: 173
    embed_dim: 64
    num_heads: 4
    max_seq_len: 5
    dropout: 0.25
  seq_model:
    model_type: "ResAttSeqEnc"
    input_dim: 64
    embed_dim: 64
    output_dim: 64
    num_heads: 4
    max_seq_len: 5
    dropout: 0.25
  optimizer:
    type: "Adam"
    lr: 0.0005
    weight_decay: 0.00005
  lr_scheduler:
    type: "ReduceLROnPlateau"
    mode: "max"
    patience: 3
""")
        critic_config_dict = config['critic']
        critic_config_dict.pop('optimizer')
        if 'lr_scheduler' in critic_config_dict:
            critic_config_dict.pop('lr_scheduler')
        self.critic_config = CriticConfig(**critic_config_dict)
        self.critic = self.critic_config.get_critic()
        self.assertIsInstance(self.critic, SeqQMixer)

    def test_get_group_consensus_mixer(self):
        config = yaml.safe_load("""
critic:
  type: "GroupConsensusMixer"
  feature_extractor:
    model_type: "Custom"
    layers:
    - type: Linear
      in_features: 54
      out_features: 32
  consensus_processor:
    model_type: "HyperNetwork"
    cond_dim: 32
    layer_dims: [24, 32, 32]
    cond_hidden_dim: 64
  model:
    model_type: "QMixModel"
    state_shape: 32
    input_dim: 3
    qmix_hidden_dim: 128
  num_agents: 3
  group_latent_dim: 8
  deterministic_eval: true
  optimizer:
    type: "Adam"
    lr: 0.0005
    weight_decay: 0.0001
""")
        critic_config_dict = config['critic']
        critic_config_dict.pop('optimizer')
        if 'lr_scheduler' in critic_config_dict:
            critic_config_dict.pop('lr_scheduler')
        self.critic_config = CriticConfig(**critic_config_dict)
        self.critic = self.critic_config.get_critic()
        self.assertIsInstance(self.critic, GroupConsensusMixer)


if __name__ == '__main__':
    unittest.main()
