import unittest
import os
import tempfile
from unittest.mock import patch, mock_open
import yaml
import torch
from src.algorithm.model.model_config import ModelConfig

class TestModelConfig(unittest.TestCase):

    def setUp(self):
        # 定义一个示例的 yaml 配置字符串
        self.yaml_config = """
        model_type: "RNN"
        input_shape: 64
        rnn_hidden_dim: 128
        rnn_layers: 1
        output_shape: 13
        """
        # 将 yaml 字符串解析为字典
        self.config_dict = yaml.safe_load(self.yaml_config)

    def test_get_model_with_pretrained_params(self):
        model_config = ModelConfig(**self.config_dict)
        model = model_config.get_model()
        # 手动填充随机参数
        with torch.no_grad():  # 禁用梯度计算
            # 使用 torch.rand 生成均匀分布的随机权重
            model.fc1.weight.data = torch.rand(model.fc1.weight.shape)
            # 使用 torch.randn 生成标准正态分布的随机偏置
            model.fc1.bias.data = torch.randn(model.fc1.bias.shape)
            # 使用 torch.rand 生成均匀分布的随机权重
            model.fc2.weight.data = torch.rand(model.fc2.weight.shape)
            # 使用 torch.randn 生成标准正态分布的随机偏置
            model.fc2.bias.data = torch.randn(model.fc2.bias.shape)

        # 模拟 torch.load 返回的模型参数
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            torch.save(model.state_dict(), tmp_file.name)
            pretrained_params_path = tmp_file.name

        # 定义 YAML 配置字符串
        yaml_config = f"""
        model_type: "RNN"
        input_shape: 64
        rnn_hidden_dim: 128
        rnn_layers: 1
        output_shape: 13
        pretrained_params_path: {pretrained_params_path}
        """

        # 解析 YAML 配置
        config_dict = yaml.safe_load(yaml_config)

        # 创建 ModelConfig 实例
        model_config = ModelConfig(**config_dict)

        # 获取模型并加载预训练参数
        loaded_model = model_config.get_model()
        # 清理临时文件
        os.remove(pretrained_params_path)

        # 检查模型是否正确加载
        for key in model.state_dict():
            self.assertTrue(torch.allclose(model.state_dict()[key], loaded_model.state_dict()[key]))

    @patch("torch.load")
    @patch("builtins.open", new_callable=mock_open, read_data="dummy_data")
    def test_get_model_without_pretrained_params(self, mock_file, mock_torch_load):
        # 创建 ModelConfig 实例
        model_config = ModelConfig(**self.config_dict)

        # 调用 get_model 方法
        model = model_config.get_model()

        # 验证模型是否正确加载
        self.assertIsNotNone(model)
        mock_file.assert_not_called()
        mock_torch_load.assert_not_called()

    @patch("torch.load")
    @patch("builtins.open", new_callable=mock_open, read_data="dummy_data")
    def test_get_model_with_invalid_model_type(self, mock_file, mock_torch_load):
        # 使用无效的模型类型
        self.config_dict["model_type"] = "InvalidModel"

        # 创建 ModelConfig 实例
        with self.assertRaises(ValueError):
            model_config = ModelConfig(**self.config_dict)
            model_config.get_model()

if __name__ == "__main__":
    unittest.main()