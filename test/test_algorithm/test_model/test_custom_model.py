import unittest
import torch
from marlite.algorithm.model.custom_model import CustomModel

class TestCustomModel(unittest.TestCase):
    def test_linear_layer(self):
        config = {
            'layers': [
                {'type': 'Linear', 'in_features': 10, 'out_features': 20}
            ]
        }
        model = CustomModel(**config)
        x = torch.randn(3, 10)  # Batch size of 3
        output = model(x)
        self.assertEqual(output.shape, (3, 20))

    def test_conv2d_layer(self):
        config = {
            'layers': [
                {'type': 'Conv2d', 'in_channels': 3, 'out_channels': 16, 'kernel_size': 5}
            ]
        }
        model = CustomModel(**config)
        x = torch.randn(3, 3, 32, 32)  # Batch size of 3, input channels of 3, spatial dimensions of 32x32
        output = model(x)
        self.assertEqual(output.shape, (3, 16, 28, 28))  # Output shape after applying a 5x5 kernel

    def test_maxpool2d_layer(self):
        config = {
            'layers': [
                {'type': 'MaxPool2d', 'kernel_size': 2}
            ]
        }
        model = CustomModel(**config)
        x = torch.randn(3, 3, 16, 16)  # Batch size of 3, input channels of 3, spatial dimensions of 16x16
        output = model(x)
        self.assertEqual(output.shape, (3, 3, 8, 8))  # Output shape after max pooling

    def test_avgpool2d_layer(self):
        config = {
            'layers': [
                {'type': 'AvgPool2d', 'kernel_size': 2}
            ]
        }
        model = CustomModel(**config)
        x = torch.randn(3, 3, 16, 16)  # Batch size of 3, input channels of 3, spatial dimensions of 16x16
        output = model(x)
        self.assertEqual(output.shape, (3, 3, 8, 8))  # Output shape after avg pooling

    def test_dropout_layer(self):
        config = {
            'layers': [
                {'type': 'Dropout', 'p': 0.5}
            ]
        }
        model = CustomModel(**config)
        x = torch.randn(3, 10)  # Batch size of 3
        output = model(x)
        self.assertEqual(output.shape, (3, 10))

    def test_batchnorm1d_layer(self):
        config = {
            'layers': [
                {'type': 'BatchNorm1d', 'num_features': 10}
            ]
        }
        model = CustomModel(**config)
        x = torch.randn(3, 10)  # Batch size of 3
        output = model(x)
        self.assertEqual(output.shape, (3, 10))

    def test_batchnorm2d_layer(self):
        config = {
            'layers': [
                {'type': 'BatchNorm2d', 'num_features': 3}
            ]
        }
        model = CustomModel(**config)
        x = torch.randn(3, 3, 16, 16)  # Batch size of 3, input channels of 3
        output = model(x)
        self.assertEqual(output.shape, (3, 3, 16, 16))

    def test_layernorm_layer(self):
        config = {
            'layers': [
                {'type': 'LayerNorm', 'normalized_shape': [3, 16, 16]}
            ]
        }
        model = CustomModel(**config)
        x = torch.randn(3, 3, 16, 16)  # Batch size of 3, input channels of 3
        output = model(x)
        self.assertEqual(output.shape, (3, 3, 16, 16))

    def test_leakyrelu_layer(self):
        config = {
            'layers': [
                {'type': 'LeakyReLU', 'negative_slope': 0.2}
            ]
        }
        model = CustomModel(**config)
        x = torch.randn(3, 10)  # Batch size of 3
        output = model(x)
        self.assertEqual(output.shape, (3, 10))

    def test_relu_layer(self):
        config = {
            'layers': [
                {'type': 'ReLU'}
            ]
        }
        model = CustomModel(**config)
        x = torch.randn(3, 10)  # Batch size of 3
        output = model(x)
        self.assertEqual(output.shape, (3, 10))

    def test_softmax_layer(self):
        config = {
            'layers': [
                {'type': 'Softmax', 'dim': 1}
            ]
        }
        model = CustomModel(**config)
        x = torch.randn(3, 10)  # Batch size of 3
        output = model(x)
        self.assertEqual(output.shape, (3, 10))

    def test_mixed_model(self):
        config = {
            'layers': [
                {'type': 'Linear', 'in_features': 10, 'out_features': 20},
                {'type': 'ReLU'},
                {'type': 'Linear', 'in_features': 20, 'out_features': 5},
                {'type': 'Softmax', 'dim': 1}
            ]
        }
        model = CustomModel(**config)
        x = torch.randn(3, 10)  # Batch size of 3
        output = model(x)
        self.assertEqual(output.shape, (3, 5))

if __name__ == '__main__':
    unittest.main()