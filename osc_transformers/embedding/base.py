import torch
import torch.nn as nn


class Embedding(nn.Module):
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError
