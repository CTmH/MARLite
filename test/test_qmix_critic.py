import unittest
import torch
import torch.nn.functional as F
from src.algorithm.critic.qmix_critic import QMIXCritic

class TestQMIXCritic(unittest.TestCase):
    def setUp(self):
        # Initialize the critic with some dummy parameters
        state_shape = 32
        input_dim = 5
        qmix_hidden_dim = 128
        hyper_hidden_dim = 64
        self.critic = QMIXCritic(state_shape, input_dim, qmix_hidden_dim, hyper_hidden_dim)

    def test_forward_pass(self):
        # Create dummy input data
        batch_size = 10
        q_values = torch.randn(batch_size, self.critic.input_dim)
        states = torch.randn(batch_size, self.critic.state_shape)

        # Perform a forward pass
        q_total = self.critic(q_values, states)

        # Check the output shape
        expected_shape = torch.zeros(batch_size).size()
        self.assertEqual(q_total.size(), expected_shape)

    def test_hyper_networks(self):
        # Test if hyper networks produce the correct shapes
        batch_size = 20
        states = torch.randn(batch_size, self.critic.state_shape)

        w1 = self.critic.hyper_w1(states)
        b1 = self.critic.hyper_b1(states)
        w2 = self.critic.hyper_w2(states)
        b2 = self.critic.hyper_b2(states)

        # Check shapes
        self.assertEqual(w1.shape, (batch_size, self.critic.input_dim * self.critic.qmix_hidden_dim))
        self.assertEqual(b1.shape, (batch_size, self.critic.qmix_hidden_dim))
        self.assertEqual(w2.shape, (batch_size, self.critic.qmix_hidden_dim))
        self.assertEqual(b2.shape, (batch_size, 1))

if __name__ == '__main__':
    unittest.main()