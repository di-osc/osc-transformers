import torch.nn as nn
import torch


class Embedding(nn.Module):
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError
