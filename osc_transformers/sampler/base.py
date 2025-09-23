import torch.nn as nn
import torch
from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_generate_tokens: int = 2048


class Sampler(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
