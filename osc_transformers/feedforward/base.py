import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
