from ..config import registry
import torch.nn as nn



@registry.layers.register("gate_mlp")
class GateMLP(nn.Module):
    def __init__(self, 
                 n_in: int, 
                 intermediate_size: int,
                 up_bias: bool = False,
                 gate_bias: bool = False,
                 down_bias: bool = False) -> None:
        super().__init__()
        self.up = nn.Linear(n_in, intermediate_size, bias=up_bias)
        self.gate = nn.Linear(n_in, intermediate_size, bias=gate_bias)
        self.down = nn.Linear(intermediate_size, n_in, bias=down_bias)
        
    def forward(self, x):
        x1 = self.up(x)
        x2 = self.gate(x)
        x = x1 * nn.functional.silu(x2)
        x = self.down(x)
        return x