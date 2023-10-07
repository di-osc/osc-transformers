import torch.nn as nn
import torch
from typing import Optional, Tuple
from ..config import registry
import math
from lightning_utilities.core.imports import RequirementCache

FlashAttention2Available = RequirementCache("flash-attn>=2.0.0.post1")
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
    
    
@registry.layers.register("causal_attention.rope")  
class RoPECaulsalAttention(nn.Module):
    """融合了RoPE,多头因果注意力机制,兼容分组注意力查询"""
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
                 dim: int, 
                 n_heads: int,
                 n_query_groups: Optional[int] = None,
                 q_bias: bool = False,
                 k_bias: bool = False,
                 v_bias: bool = False,
                 o_bias: bool = False) -> None:
        super().__init__()
        assert dim % n_heads == 0, f"dim {dim} must be divisible by n_heads {n_heads}"
        self.dim = dim
        self.n_heads = n_heads
        self.head_size = dim // n_heads
        self.n_query_groups = n_query_groups if n_query_groups else n_heads
        self.q_proj = nn.Linear(self.dim, self.n_heads * self.head_size, bias=q_bias)
        self.k_proj = nn.Linear(self.dim, self.n_query_groups * self.head_size, bias=k_bias)
        self.v_proj = nn.Linear(self.dim, self.n_query_groups * self.head_size, bias=v_bias)
        self.o_proj = nn.Linear(self.dim, self.n_query_groups * self.head_size, bias=o_bias)
        
        self.kv_cache: Optional[KVCache] = None


    def forward(self,
                x: torch.Tensor,
                cos: torch.Tensor,
                sin: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                input_pos: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[KVCache]]:
        
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_size).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, self.n_query_groups, self.head_size).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.n_query_groups, self.head_size).permute(0, 2, 1, 3)

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
        if (
            FlashAttention2Available
            and mask is None
            and q.device.type == "cuda"
            and q.dtype in (torch.float16, torch.bfloat16)
        ):
            from flash_attn import flash_attn_func

            # flash-attn requires (B, T, nh, hs)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            return flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=scale, causal=True)
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