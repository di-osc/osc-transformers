import torch
from ..config import registry


@registry.layers.register("RMSNorm")
class RMSNorm(torch.nn.Module):
    def __init__(self,
                 n_in: int,
                 eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(n_in))
        self.eps = eps
        
    def forward(self, x):
        norm = torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps
        x = x * torch.rsqrt(norm)
        x = x * self.weight
        return x
    
    def reset_parameters(self):
        self.weight.data.fill_(1.0)