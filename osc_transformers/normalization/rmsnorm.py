import torch
import torch.nn as nn

from .base import Normalization
from ..registry import Registry
from ..ops.rms_norm import LigerRMSNormFunction


@Registry.normalization.register("RMSNorm.compile")
class TorchRMSNorm(Normalization):
    def __init__(self, in_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(in_dim))
        self.eps = eps

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.to(torch.float32).add_(residual.to(torch.float32))
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)


@Registry.normalization.register("RMSNorm")
@Registry.normalization.register("RMSNorm.triton")
class TritonRMSNorm(Normalization):
    def __init__(
        self,
        in_dim,
        eps=1e-6,
        offset=0.0,
        casting_mode="llama",
        init_fn="ones",
        in_place=True,
        row_mode=None,
    ):
        super().__init__()
        assert init_fn in [
            "ones",
            "zeros",
        ], f"init_fn must be either 'ones' or 'zeros', got {init_fn}"
        self.weight = nn.Parameter(
            torch.ones(in_dim) if init_fn == "ones" else torch.zeros(in_dim)
        )
        (
            self.variance_epsilon,
            self.offset,
            self.casting_mode,
            self.in_place,
            self.row_mode,
        ) = (
            eps,
            offset,
            casting_mode,
            in_place,
            row_mode,
        )

    def forward(self, hidden_states: torch.Tensor):
        return LigerRMSNormFunction.apply(
            hidden_states,
            self.weight,
            self.variance_epsilon,
            self.offset,
            self.casting_mode,
            self.in_place,
            self.row_mode,
        )
