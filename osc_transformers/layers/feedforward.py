from ..config import registry
import torch.nn as nn


class GLU(nn.Module):
    """门控线性单元
    """
    def __init__(self, 
                 n_in: int, 
                 n_hidden: int,
                 activation: nn.Module,
                 up_bias: bool = False,
                 gate_bias: bool = False,
                 down_bias: bool = False) -> None:
        super().__init__()
        self.up_proj = nn.Linear(n_in, n_hidden, bias=up_bias)
        self.gate_proj = nn.Linear(n_in, n_hidden, bias=gate_bias)
        self.down_proj = nn.Linear(n_hidden, n_in, bias=down_bias)
        self.activation = activation
        
    def forward(self, x):
        x1 = self.up_proj(x)
        x2 = self.gate_proj(x)
        x = x1 * self.activation(x2)
        x = self.down_proj(x)
        return x
    
    
    
@registry.layers.register("SwiGLU")
def SwiGLU(n_in: int, 
           n_hidden: int,
           up_bias: bool = False,
           gate_bias: bool = False,
           down_bias: bool = False) -> nn.Module:
    """Swish激活函数的门控线性单元
    """
    return GLU(n_in=n_in,
               n_hidden=n_hidden,
               activation=nn.SiLU(),
               up_bias=up_bias,
               gate_bias=gate_bias,
               down_bias=down_bias)
    
    
@registry.layers.register("GeGLU")
def GeGLU(n_in: int, 
          n_hidden: int,
          up_bias: bool = False,
          gate_bias: bool = False,
          down_bias: bool = False) -> nn.Module:
    """GELU激活函数的门控线性单元
    """
    return GLU(n_in=n_in,
               n_hidden=n_hidden,
               activation=nn.GELU(),
               up_bias=up_bias,
               gate_bias=gate_bias,
               down_bias=down_bias)