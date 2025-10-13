import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ops.swiglu import LigerSiLUMulFunction
from ..registry import Registry
from .base import FeedForward


@Registry.feedforward.register("SwiGLU")
@Registry.feedforward.register("SwiGLU.torch")
class SwiGLU(FeedForward):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        up_bias: bool = False,
        gate_bias: bool = False,
        down_bias: bool = False,
    ):
        super().__init__()
        self.up_proj = nn.Linear(in_dim, hidden_dim, bias=up_bias)
        self.gate_proj = nn.Linear(in_dim, hidden_dim, bias=gate_bias)
        self.down_proj = nn.Linear(hidden_dim, in_dim, bias=down_bias)

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(h)
        return x


@Registry.feedforward.register("SwiGLU.triton")
class TritonSwiGLU(FeedForward):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        up_bias: bool = False,
        gate_bias: bool = False,
        down_bias: bool = False,
    ):
        super().__init__()
        self.up_proj = nn.Linear(in_dim, hidden_dim, bias=up_bias)
        self.gate_proj = nn.Linear(in_dim, hidden_dim, bias=gate_bias)
        self.down_proj = nn.Linear(hidden_dim, in_dim, bias=down_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(LigerSiLUMulFunction.apply(self.gate_proj(x), self.up_proj(x)))
