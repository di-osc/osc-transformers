import torch.nn as nn
import torch

from ..registry import Registry
from .base import Head


@Registry.head.register("LMHead")
class LMHead(Head):
    def __init__(self, in_dim: int, out_dim: int, bias: bool) -> None:
        super().__init__()
        self.predictor = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(x)
