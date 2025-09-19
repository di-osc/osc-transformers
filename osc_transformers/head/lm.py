import torch.nn as nn
import torch

from ..registry import Registry
from .base import Head


@Registry.head.register("LMHead")
class LMHead(Head):
    def __init__(self, in_dim: int, out_dim: int, bias: bool) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bias = bias
        self.predictor = nn.Linear(
            in_features=self.in_dim, out_features=self.out_dim, bias=self.bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(x)
