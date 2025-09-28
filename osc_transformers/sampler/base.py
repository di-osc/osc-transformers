from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_generate_tokens: int = 2048
    top_p: float = 0.75
    top_k: int = 100
    repetition_penalty: float = 1.0


class Sampler(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
