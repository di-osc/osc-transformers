import torch.nn as nn
import torch
import torch.nn.functional as F
from ..config import registry



@registry.layers.register("Linear")
def Linear(n_in: int, n_out: int, bias: bool = True):
    return nn.Linear(in_features=n_in, out_features=n_out, bias=bias)


@registry.layers.register("WeightOnlyInt8Linear")
class WeightOnlyInt8Linear(nn.Module):
    """用于量化的Linear层，只包含weight和scales两个参数"""
    __contains__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device = None,
        dtype = None,
    ):
        # factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.register_buffer("weight", torch.empty((out_features, in_features), dtype=torch.int8))
        self.register_buffer("scales", torch.ones(out_features, dtype=torch.bfloat16))
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.to(dtype=input.dtype), bias=self.bias) * self.scales
    

@registry.layers.register("WeightOnlyInt4Linear")
class WeightOnlyInt4Linear(torch.nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
            self, 
            in_features: int, 
            out_features: int,
            bias=True, 
            device=None, 
            dtype=None, 
            groupsize: int = 128, 
            inner_k_tiles: int = 8, 
            use_cuda=True,
    ) -> None:
        super().__init__()
        self.padding = not self._check_linear_int4_k(in_features, groupsize, inner_k_tiles)
        if self.padding:
            self.origin_in_features = in_features
            in_features = find_multiple(in_features, 1024)

        self.in_features = in_features
        self.out_features = out_features
        assert not bias, "require bias=False"
        self.groupsize = groupsize
        self.inner_k_tiles = inner_k_tiles

        assert out_features % 8 == 0, "require out_features % 8 == 0"
        assert in_features % (inner_k_tiles * 16) == 0, "require in_features % (innerKTiles * 16) == 0"
        self.register_buffer(
            "weight",
            torch.empty((out_features // 8, in_features // (inner_k_tiles * 16), 32, inner_k_tiles // 2), dtype=torch.int32)
        )
        self.register_buffer(
            "scales_and_zeros",
            torch.empty((in_features // groupsize, out_features, 2), dtype=torch.bfloat16)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.to(torch.bfloat16)
        if self.padding:
            import torch.nn.functional as F
            input = F.pad(input, pad=(0, self.in_features - self.origin_in_features))
        return self._linear_forward_int4(input,
                                        self.weight, 
                                        self.scales_and_zeros, 
                                        self.out_features, 
                                        self.groupsize)
        
    def _check_linear_int4_k(self, in_features: int, groupsize: int, inner_k_tiles: int) -> bool:
        """check if the input features are compatible with the linear int4 kernel

        Args:
            in_features (int): the number of input features
            groupsize (int): the group size
            inner_k_tiles (int): the number of inner k tiles
        """
        return in_features % (inner_k_tiles * 16) == 0 and in_features % groupsize == 0
    
    def _linear_forward_int4(self, x, weight_int4pack, scales_and_zeros, out_features, groupsize):
        origin_x_size = x.size()
        x = x.reshape(-1, origin_x_size[-1])
        c = torch.ops.aten._weight_int4pack_mm(x, weight_int4pack, groupsize, scales_and_zeros)
        new_shape = origin_x_size[:-1] + (out_features,)
        c = c.reshape(new_shape)
        return c
    

def find_multiple(n: int, k: int) -> int:
    """Find the smallest multiple of k that is greater than or equal to n.

    Args:
        n (int): the number to find the multiple of k for.
        k (int): the number to find the multiple of.

    Returns:
        int: the smallest multiple of k that is greater than or equal to n.
    """
    if n % k == 0:
        return n
    return n + k - (n % k)