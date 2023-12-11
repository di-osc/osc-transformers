import torch.nn as nn
import torch
from typing import Optional, Tuple
from ..config import registry
import math


RoPECache = Tuple[torch.Tensor, torch.Tensor]

class KVCache(nn.Module):
    def __init__(self, 
                 k_shape: Tuple[int, int, int, int],
                 v_shape: Tuple[int, int, int, int],
                 device: torch.device,
                 dtype: torch.dtype) -> None:
        super().__init__()
        self.register_buffer("k", torch.zeros(k_shape, device=device, dtype=dtype), persistent=False)
        self.register_buffer("v", torch.zeros(v_shape, device=device, dtype=dtype), persistent=False)
        
    def forward(self, k: torch.Tensor, v: torch.Tensor, input_pos: torch.Tensor):
        
        self.k = self.k.to(k.dtype)
        self.v = self.v.to(v.dtype)
        
        k = self.k.index_copy_(2, input_pos, k)
        v = self.v.index_copy_(2, input_pos, v)
        return k, v
    
    
@registry.layers.register("SelfAttention.v1")  
class SelfAttention(nn.Module):
    """自注意力机制融合了RoPE,多头注意力,分组注意力"""
    # 以 `n_head=4`举例说明:
    # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
    # │ v ││ v ││ v ││ v │     │ v │    │ v │             │ v │
    # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
    #   │    │    │    │         │        │                 │
    # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
    # │ k ││ k ││ k ││ k │     │ k │    │ k │             │ k │
    # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
    #   │    │    │    │      ┌──┴──┐  ┌──┴──┐      ┌────┬──┴─┬────┐
    # ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐
    # │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │
    # └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘
    # ◀──────────────────▶  ◀──────────────────▶  ◀──────────────────▶
    #         MHA                    GQA                   MQA
    #   n_query_groups=4       n_query_groups=2      n_query_groups=1
    
    def __init__(self, 
                 n_in: int, 
                 n_heads: int,
                 n_query_groups: Optional[int] = None,
                 q_bias: bool = False,
                 k_bias: bool = False,
                 v_bias: bool = False,
                 o_bias: bool = False) -> None:
        super().__init__()
        assert n_in % n_heads == 0, f"dim {n_in} must be divisible by n_heads {n_heads}"
        self.n_heads = n_heads
        self.head_size = n_in // n_heads
        self.n_query_groups = n_query_groups if n_query_groups else n_heads
        self.q_proj = nn.Linear(n_in, self.n_heads * self.head_size, bias=q_bias)
        self.k_proj = nn.Linear(n_in, self.n_query_groups * self.head_size, bias=k_bias)
        self.v_proj = nn.Linear(n_in, self.n_query_groups * self.head_size, bias=v_bias)
        self.o_proj = nn.Linear(n_in, n_in, bias=o_bias)
        
        self.kv_cache: Optional[KVCache] = None


    def forward(self,
                x: torch.Tensor,
                cos: Optional[torch.Tensor] = None,
                sin: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                input_pos: Optional[torch.Tensor] = None,
                **kwargs) -> Tuple[torch.Tensor, Optional[KVCache]]:
        
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        q: torch.Tensor = self.q_proj(x).reshape(B, T, self.n_heads, self.head_size).permute(0, 2, 1, 3)
        k: torch.Tensor = self.k_proj(x).reshape(B, T, self.n_query_groups, self.head_size).permute(0, 2, 1, 3)
        v: torch.Tensor = self.v_proj(x).reshape(B, T, self.n_query_groups, self.head_size).permute(0, 2, 1, 3)
        
        if (cos is not None) and (sin is not None):
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
            
        # repeat k and v if necessary
        if self.n_query_groups != 1 and self.n_query_groups != self.n_heads:  # doing this would require a full kv cache with MQA (inefficient!)
            # for MHA this is a no-op
            k = k[:,:,None,:,:].expand(-1, -1, self.n_heads // self.n_query_groups, -1, -1).reshape(B, self.n_heads, T, self.head_size)
            v = v[:,:,None,:,:].expand(-1, -1, self.n_heads // self.n_query_groups, -1, -1).reshape(B, self.n_heads, T, self.head_size)
        
        if input_pos is not None:
            if not isinstance(self.kv_cache, KVCache):
                raise TypeError("You need to call `model.build_kv_caches()`")
            k, v = self.kv_cache(input_pos=input_pos, k=k, v=v)

        y = self.scaled_dot_product_attention(q, k, v, mask=mask)

        y = y.reshape(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.o_proj(y)

        return y

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ):
        scale = 1.0 / math.sqrt(self.head_size)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None
        )
        return y.transpose(1, 2)
    
    def build_kv_cache(self, 
                     batch_size: int, 
                     max_seq_length: int, 
                     device: Optional[torch.device] = None, 
                     dtype: Optional[torch.dtype] = None) -> None:
        n_heads = 1 if self.n_query_groups == 1 else self.n_heads
        k_shape = (batch_size, n_heads, max_seq_length, self.head_size)
        v_shape = (batch_size, n_heads, max_seq_length, self.head_size)
        self.kv_cache = KVCache(k_shape, v_shape, device, dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_size = x.size(-1)
    x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
    x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
    roped = (x * cos) + (rotated * sin)
    return roped.type_as(x) 