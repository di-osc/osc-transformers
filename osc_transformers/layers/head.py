import torch.nn as nn
from ..config import registry
import torch
from typing import Optional



@registry.layers.register("lm_head")
class LMHead(nn.Module):
    def __init__(self, 
                 norm: nn.Module,
                 classifier: nn.Module) -> None:
        super().__init__()
        self.norm = norm
        self.classfier = classifier
        
    def forward(self, x):
        x = self.norm(x)
        x = self.classfier(x)
        return x
    

@registry.layers.register("linear")
def build_linear_layer(n_in: int, n_out: int, bias: bool = True, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None):
    return nn.Linear(n_in, n_out, bias=bias, dtype=dtype, device=device)