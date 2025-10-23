import pytest
import torch

from osc_transformers.feedforward.swiglu import SwiGLU, TritonSwiGLU


class TestSwiGLU:
    """测试 SwiGLU 和 TritonSwiGLU 的输出一致性"""

    @pytest.fixture
    def setup_models(self):
        """设置测试用的模型参数"""
        in_dim = 64
        hidden_dim = 128
        batch_size = 4
        seq_len = 32

        # 创建两个模型，参数完全一致
        torch_model = SwiGLU(in_dim=in_dim, hidden_dim=hidden_dim).cuda()
        triton_model = TritonSwiGLU(in_dim=in_dim, hidden_dim=hidden_dim).cuda()

        # 复制权重以确保完全一致
        triton_model.up_proj.load_state_dict(torch_model.up_proj.state_dict())
        triton_model.gate_proj.load_state_dict(torch_model.gate_proj.state_dict())
        triton_model.down_proj.load_state_dict(torch_model.down_proj.state_dict())

        # 创建测试输入
        x = torch.randn(batch_size, seq_len, in_dim).cuda()

        return torch_model, triton_model, x

    def test_output_consistency_no_bias(self, setup_models):
        """测试无偏置情况下的输出一致性"""
        torch_model, triton_model, x = setup_models

        # 设置为无偏置
        torch_model.up_proj.bias = None
        torch_model.gate_proj.bias = None
        torch_model.down_proj.bias = None

        triton_model.up_proj.bias = None
        triton_model.gate_proj.bias = None
        triton_model.down_proj.bias = None

        with torch.no_grad():
            torch_output = torch_model(x)
            triton_output = triton_model(x)

        # 检查输出形状一致
        assert torch_output.shape == triton_output.shape, f"输出形状不一致: torch {torch_output.shape} vs triton {triton_output.shape}"

        # 检查输出值一致（允许小数值差异）
        torch.testing.assert_close(torch_output, triton_output, rtol=1e-5, atol=1e-6)

    def test_output_consistency_with_bias(self, setup_models):
        """测试有偏置情况下的输出一致性"""
        torch_model, triton_model, x = setup_models

        # 设置为有偏置
        torch_model.up_proj.bias = torch.nn.Parameter(torch.randn(torch_model.up_proj.out_features).cuda())
        torch_model.gate_proj.bias = torch.nn.Parameter(torch.randn(torch_model.gate_proj.out_features).cuda())
        torch_model.down_proj.bias = torch.nn.Parameter(torch.randn(torch_model.down_proj.out_features).cuda())

        triton_model.up_proj.bias = torch.nn.Parameter(torch_model.up_proj.bias.clone())
        triton_model.gate_proj.bias = torch.nn.Parameter(torch_model.gate_proj.bias.clone())
        triton_model.down_proj.bias = torch.nn.Parameter(torch_model.down_proj.bias.clone())

        with torch.no_grad():
            torch_output = torch_model(x)
            triton_output = triton_model(x)

        # 检查输出形状一致
        assert torch_output.shape == triton_output.shape, f"输出形状不一致: torch {torch_output.shape} vs triton {triton_output.shape}"

        # 检查输出值一致（允许小数值差异）
        torch.testing.assert_close(torch_output, triton_output, rtol=1e-5, atol=1e-6)

    def test_different_input_shapes(self):
        """测试不同输入形状下的输出一致性"""
        in_dim = 32
        hidden_dim = 64

        torch_model = SwiGLU(in_dim=in_dim, hidden_dim=hidden_dim).cuda()
        triton_model = TritonSwiGLU(in_dim=in_dim, hidden_dim=hidden_dim).cuda()

        # 复制权重
        triton_model.up_proj.load_state_dict(torch_model.up_proj.state_dict())
        triton_model.gate_proj.load_state_dict(torch_model.gate_proj.state_dict())
        triton_model.down_proj.load_state_dict(torch_model.down_proj.state_dict())

        # 测试不同形状的输入
        test_shapes = [
            (2, 16, in_dim),  # (batch, seq, dim)
            (1, 8, in_dim),
            (8, 4, in_dim),
        ]

        for shape in test_shapes:
            x = torch.randn(*shape).cuda()

            with torch.no_grad():
                torch_output = torch_model(x)
                triton_output = triton_model(x)

            # 检查输出形状一致
            assert torch_output.shape == triton_output.shape, f"形状 {shape} 输出形状不一致"

            # 检查输出值一致
            torch.testing.assert_close(torch_output, triton_output, rtol=1e-5, atol=1e-6)

    def test_gradient_consistency(self):
        """测试梯度计算的一致性"""
        in_dim = 32
        hidden_dim = 64

        torch_model = SwiGLU(in_dim=in_dim, hidden_dim=hidden_dim).cuda()
        triton_model = TritonSwiGLU(in_dim=in_dim, hidden_dim=hidden_dim).cuda()

        # 复制权重
        triton_model.up_proj.load_state_dict(torch_model.up_proj.state_dict())
        triton_model.gate_proj.load_state_dict(torch_model.gate_proj.state_dict())
        triton_model.down_proj.load_state_dict(torch_model.down_proj.state_dict())

        x = torch.randn(2, 8, in_dim).cuda()
        x.requires_grad_(True)

        # 前向传播
        torch_output = torch_model(x)
        triton_output = triton_model(x)

        # 检查前向输出一致
        torch.testing.assert_close(torch_output, triton_output, rtol=1e-5, atol=1e-6)

        # 反向传播
        torch_output.sum().backward()
        torch_grad = x.grad.clone()

        x.grad.zero_()

        triton_output.sum().backward()
        triton_grad = x.grad.clone()

        # 检查梯度一致
        torch.testing.assert_close(torch_grad, triton_grad, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])
