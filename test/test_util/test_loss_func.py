import unittest
import torch
import torch.nn.functional as F
from marlite.util.loss_func import ChamferDistanceLoss

class TestChamferDistanceLoss(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.batch_size = 2
        self.n_points = 4
        self.feature_dim = 5

    def test_basic_functionality(self):
        """Test basic Chamfer Distance computation."""
        # Create simple test data
        pred = torch.randn(self.batch_size, self.n_points, self.feature_dim)
        target = torch.randn(self.batch_size, self.n_points, self.feature_dim)

        # Create loss function
        loss_fn = ChamferDistanceLoss(reduction='mean')

        # Compute loss
        loss = loss_fn(pred, target)

        # Check that loss is a scalar tensor
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Scalar tensor

    def test_reduction_none(self):
        """Test reduction='none' returns correct shape."""
        pred = torch.randn(self.batch_size, self.n_points, self.feature_dim)
        target = torch.randn(self.batch_size, self.n_points, self.feature_dim)

        loss_fn = ChamferDistanceLoss(reduction='none')
        loss = loss_fn(pred, target)

        # Should return (batch_size,) tensor
        self.assertEqual(loss.shape, (self.batch_size,))

    def test_reduction_sum(self):
        """Test reduction='sum' computes correct sum."""
        pred = torch.randn(self.batch_size, self.n_points, self.feature_dim)
        target = torch.randn(self.batch_size, self.n_points, self.feature_dim)

        loss_fn = ChamferDistanceLoss(reduction='sum')
        loss = loss_fn(pred, target)

        # Should be a scalar tensor
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)

    def test_with_mask(self):
        """Test Chamfer Distance with masking."""
        pred = torch.randn(self.batch_size, self.n_points, self.feature_dim)
        target = torch.randn(self.batch_size, self.n_points, self.feature_dim)

        # Create mask where some entries are invalid (False)
        mask = torch.ones(self.batch_size, self.n_points, dtype=torch.bool)
        mask[0, 2:] = False  # Mark last 2 points as invalid for first batch

        loss_fn = ChamferDistanceLoss(reduction='mean')
        loss = loss_fn(pred, target, mask)

        # Should still return a scalar
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)

    def test_identical_tensors(self):
        """Test that identical tensors give zero loss."""
        tensor = torch.randn(self.batch_size, self.n_points, self.feature_dim)

        loss_fn = ChamferDistanceLoss(reduction='mean')
        loss = loss_fn(tensor, tensor)

        # Loss should be very close to zero
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_zero_tensors(self):
        """Test with zero tensors."""
        pred = torch.zeros(self.batch_size, self.n_points, self.feature_dim)
        target = torch.zeros(self.batch_size, self.n_points, self.feature_dim)

        loss_fn = ChamferDistanceLoss(reduction='mean')
        loss = loss_fn(pred, target)

        # Loss should be exactly zero
        self.assertEqual(loss.item(), 0.0)

    def test_different_tensors(self):
        """Test with clearly different tensors."""
        pred = torch.ones(self.batch_size, self.n_points, self.feature_dim)
        target = torch.zeros(self.batch_size, self.n_points, self.feature_dim)

        loss_fn = ChamferDistanceLoss(reduction='mean')
        loss = loss_fn(pred, target)

        # Loss should be positive
        self.assertGreater(loss.item(), 0.0)

    def test_large_difference(self):
        """Test with large differences between tensors."""
        pred = torch.full((self.batch_size, self.n_points, self.feature_dim), 10.0)
        target = torch.full((self.batch_size, self.n_points, self.feature_dim), 1.0)

        loss_fn = ChamferDistanceLoss(reduction='mean')
        loss = loss_fn(pred, target)

        # Loss should be large
        self.assertGreater(loss.item(), 10.0)

    def test_computational_correctness(self):
        """Test that Chamfer Distance computation is mathematically correct for simple case."""
        # Simple 2D case
        batch_size, n_points, feature_dim = 1, 2, 2

        # Two points: (0,0) and (1,1)
        pred = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]], dtype=torch.float32)
        # Two points: (0,1) and (1,0)
        target = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], dtype=torch.float32)

        loss_fn = ChamferDistanceLoss(reduction='none', use_squared_distance=True)
        loss = loss_fn(pred, target)

        # For each point in pred, find closest point in target
        # Point (0,0) in pred -> closest point (0,1) in target -> squared distance = 1
        # Point (1,1) in pred -> closest point (1,0) in target -> squared distance = 1
        # Forward: (1 + 1) / 2 = 1
        # Backward: (1 + 1) / 2 = 1
        # Total: 1 + 1 = 2

        expected_loss = torch.tensor([2.0])  # Expected Chamfer distance

        # Check that loss is computed correctly (allowing for small numerical errors)
        self.assertAlmostEqual(loss.item(), 2.0, places=5)

    def test_backward_pass(self):
        """Test that gradients can be computed through the loss."""
        pred = torch.randn(self.batch_size, self.n_points, self.feature_dim, requires_grad=True)
        target = torch.randn(self.batch_size, self.n_points, self.feature_dim)

        loss_fn = ChamferDistanceLoss(reduction='mean')
        loss = loss_fn(pred, target)

        # Compute gradients
        loss.backward()

        # Check that gradients were computed
        self.assertIsNotNone(pred.grad)
        self.assertTrue(pred.grad.shape == pred.shape)

    def test_squared_distance_false(self):
        """Test Chamfer Distance with non-squared distance."""
        pred = torch.randn(self.batch_size, self.n_points, self.feature_dim)
        target = torch.randn(self.batch_size, self.n_points, self.feature_dim)

        loss_fn = ChamferDistanceLoss(reduction='mean', use_squared_distance=False)
        loss = loss_fn(pred, target)

        # Check that loss is a scalar tensor
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)

if __name__ == '__main__':
    unittest.main()