import torch.nn as nn
import torch


class FeedForward(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
