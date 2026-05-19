import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import Registry
from .base import FeedForward


@Registry.feedforward.register("SwiGLU")
@Registry.feedforward.register("SwiGLU.torch")
@Registry.feedforward.register("SwiGLU.triton")
class SwiGLU(FeedForward):
    """SwiGLU feed-forward block with fused gate/up projection.

    The previous implementation used two separate projections:
        silu(gate_proj(x)) * up_proj(x)

    This implementation computes both projections with one wider Linear:
        gate, up = gate_up_proj(x).chunk(2, dim=-1)

    That removes one GEMM launch and one read of the input activations, which
    is more useful for inference than a standalone Triton elementwise kernel.
    The output layout is [gate, up] to keep the forward expression direct.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        up_bias: bool = False,
        gate_bias: bool = False,
        down_bias: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.has_gate_bias = gate_bias
        self.has_up_bias = up_bias

        self.gate_up_proj = nn.Linear(in_dim, hidden_dim * 2, bias=gate_bias or up_bias)
        self.down_proj = nn.Linear(hidden_dim, in_dim, bias=down_bias)

        if self.gate_up_proj.bias is not None:
            with torch.no_grad():
                if not gate_bias:
                    self.gate_up_proj.bias[:hidden_dim].zero_()
                if not up_bias:
                    self.gate_up_proj.bias[hidden_dim:].zero_()

    @torch.compile(fullgraph=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        gate_weight_key = prefix + "gate_proj.weight"
        up_weight_key = prefix + "up_proj.weight"
        fused_weight_key = prefix + "gate_up_proj.weight"
        if fused_weight_key not in state_dict and gate_weight_key in state_dict and up_weight_key in state_dict:
            state_dict[fused_weight_key] = torch.cat([state_dict[gate_weight_key], state_dict[up_weight_key]], dim=0)
            del state_dict[gate_weight_key]
            del state_dict[up_weight_key]

        fused_bias_key = prefix + "gate_up_proj.bias"
        if self.gate_up_proj.bias is not None and fused_bias_key not in state_dict:
            gate_bias_key = prefix + "gate_proj.bias"
            up_bias_key = prefix + "up_proj.bias"
            gate_bias = state_dict.pop(gate_bias_key, None)
            up_bias = state_dict.pop(up_bias_key, None)
            if gate_bias is not None or up_bias is not None:
                bias = self.gate_up_proj.bias.detach().new_zeros(self.hidden_dim * 2)
                if gate_bias is not None:
                    bias[: self.hidden_dim] = gate_bias
                if up_bias is not None:
                    bias[self.hidden_dim :] = up_bias
                state_dict[fused_bias_key] = bias

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class TritonSwiGLU(SwiGLU):
    """Backward-compatible alias for the old Triton registry/import name."""
