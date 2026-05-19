import pytest
import torch
import torch.nn.functional as F

from osc_transformers.feedforward.swiglu import SwiGLU, TritonSwiGLU


def _reference_swiglu(model: SwiGLU, x: torch.Tensor) -> torch.Tensor:
    gate_up = model.gate_up_proj(x)
    gate, up = gate_up.chunk(2, dim=-1)
    return model.down_proj(F.silu(gate) * up)


def _make_model(in_dim: int, hidden_dim: int, **kwargs) -> SwiGLU:
    model = SwiGLU(in_dim=in_dim, hidden_dim=hidden_dim, **kwargs)
    return model.cuda()


class TestSwiGLU:
    def test_forward_matches_reference(self):
        model = _make_model(in_dim=64, hidden_dim=128).eval()
        x = torch.randn(4, 32, 64, device="cuda")

        with torch.no_grad():
            actual = model(x)
            expected = _reference_swiglu(model, x)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize(
        "up_bias,gate_bias,down_bias",
        [
            (False, False, False),
            (True, False, False),
            (False, True, False),
            (True, True, True),
        ],
    )
    def test_bias_options(self, up_bias: bool, gate_bias: bool, down_bias: bool):
        model = _make_model(
            in_dim=32,
            hidden_dim=64,
            up_bias=up_bias,
            gate_bias=gate_bias,
            down_bias=down_bias,
        ).eval()
        x = torch.randn(2, 8, 32, device="cuda")

        with torch.no_grad():
            actual = model(x)
            expected = _reference_swiglu(model, x)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("shape", [(2, 16, 32), (1, 8, 32), (8, 4, 32), (64, 32)])
    def test_different_input_shapes(self, shape):
        model = _make_model(in_dim=32, hidden_dim=64).eval()
        x = torch.randn(*shape, device="cuda")

        with torch.no_grad():
            actual = model(x)
            expected = _reference_swiglu(model, x)

        assert actual.shape == (*shape[:-1], 32)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_gradient(self):
        model = _make_model(in_dim=32, hidden_dim=64)
        x = torch.randn(2, 8, 32, device="cuda", requires_grad=True)

        output = model(x)
        output.sum().backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_triton_swiglu_alias_loads_same_state(self):
        model = _make_model(in_dim=32, hidden_dim=64).eval()
        alias = TritonSwiGLU(in_dim=32, hidden_dim=64).cuda().eval()
        alias.load_state_dict(model.state_dict())
        x = torch.randn(2, 8, 32, device="cuda")

        with torch.no_grad():
            torch.testing.assert_close(alias(x), model(x), rtol=1e-5, atol=1e-6)

    def test_loads_legacy_separate_projection_state_dict(self):
        model = _make_model(in_dim=32, hidden_dim=64, up_bias=True, gate_bias=True, down_bias=True).eval()
        legacy_state = {
            "gate_proj.weight": model.gate_up_proj.weight[:64].detach().clone(),
            "up_proj.weight": model.gate_up_proj.weight[64:].detach().clone(),
            "gate_proj.bias": model.gate_up_proj.bias[:64].detach().clone(),
            "up_proj.bias": model.gate_up_proj.bias[64:].detach().clone(),
            "down_proj.weight": model.down_proj.weight.detach().clone(),
            "down_proj.bias": model.down_proj.bias.detach().clone(),
        }
        loaded = _make_model(in_dim=32, hidden_dim=64, up_bias=True, gate_bias=True, down_bias=True).eval()
        loaded.load_state_dict(legacy_state)
        x = torch.randn(2, 8, 32, device="cuda")

        with torch.no_grad():
            torch.testing.assert_close(loaded(x), model(x), rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])
