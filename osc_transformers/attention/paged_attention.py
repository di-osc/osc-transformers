import math

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..ops.attention import attn_varlen, attn_with_paged_kvcache
from ..ops.rotary import apply_rope, build_rope_cache
from ..registry import Registry
from .base import AttentionContext, CausalSelfAttention


@Registry.attention.register("PagedAttention")
class PagedAttention(CausalSelfAttention):
    """兼容MQA,GQA,MHA的因果自注意力机制层"""

    # 当`num_heads=4`时MHA,GQA,MQA的区别:
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
    #   num_query_groups=4       num_query_groups=2      num_query_groups=1

    def __init__(
        self,
        in_dim: int,
        num_heads: int,
        head_dim: int | None = None,
        q_bias: bool = False,
        k_bias: bool = False,
        v_bias: bool = False,
        o_bias: bool = False,
        num_query_groups: int | None = None,
        use_qkv_proj: bool = False,
        qkv_bias: bool = False,
        q_norm: nn.Module | None = None,
        k_norm: nn.Module | None = None,
        apply_rope: bool = True,
        rope_base: int = 10000,
        scale: float | None = None,
    ):
        super().__init__()

        assert in_dim % num_heads == 0, f"dim {in_dim} must be divisible by num_heads {num_heads}"

        self.num_heads = num_heads
        self.head_dim = head_dim or in_dim // num_heads
        self.num_query_groups = num_query_groups or num_heads

        self.use_qkv_proj = use_qkv_proj
        if not use_qkv_proj:
            self.q_proj = nn.Linear(in_dim, self.num_heads * self.head_dim, bias=q_bias)
            self.k_proj = nn.Linear(in_dim, self.num_query_groups * self.head_dim, bias=k_bias)
            self.v_proj = nn.Linear(in_dim, self.num_query_groups * self.head_dim, bias=v_bias)
        else:
            self.qkv_proj = nn.Linear(
                in_dim,
                self.num_heads * self.head_dim + self.num_query_groups * self.head_dim * 2,
                bias=qkv_bias,
            )

        self.o_proj = nn.Linear(self.num_heads * self.head_dim, in_dim, bias=o_bias)

        self.q_norm = q_norm
        self.k_norm = k_norm

        self.k_cache: torch.Tensor = torch.tensor([])
        self.v_cache: torch.Tensor = torch.tensor([])
        self.rope_sin_cache: torch.Tensor = torch.tensor([])
        self.rope_cos_cache: torch.Tensor = torch.tensor([])
        self.apply_rope = apply_rope
        self.rope_base = rope_base
        self.scale = scale or 1.0 / math.sqrt(self.head_dim)
        self.block_size = None

    def forward(
        self,
        x: torch.Tensor,
        attn_ctx: AttentionContext,
    ):
        """Forward pass for the PagedAttention layer.

        Args:
            x (torch.Tensor): Input tensor of shape (L, D) where L is the batch sequence length and D is the embedding dimensionality (n_embd).
            attn_ctx (AttentionContext): Attention context.
        """
        L, _ = x.size()

        q, k, v = self.qkv_forward(x)
        q, k, v = (
            q.reshape(L, self.num_heads, self.head_dim),
            k.reshape(L, self.num_query_groups, self.head_dim),
            v.reshape(L, self.num_query_groups, self.head_dim),
        )
        if self.q_norm is not None:
            q: torch.Tensor = self.q_norm(q)

        if self.k_norm is not None:
            k: torch.Tensor = self.k_norm(k)

        if self.apply_rope:
            cos = self.rope_cos_cache[attn_ctx.input_pos]
            sin = self.rope_sin_cache[attn_ctx.input_pos]
            # q, k = q.transpose(0, 1), k.transpose(0, 1)
            q = apply_rope(q, cos, sin, cu_seqlens=attn_ctx.cu_seqlens_q, max_seqlen=attn_ctx.max_seqlen_q)
            k = apply_rope(k, cos, sin, cu_seqlens=attn_ctx.cu_seqlens_k, max_seqlen=attn_ctx.max_seqlen_k)
            # q, k = q.transpose(0, 1), k.transpose(0, 1)

        o = self.scaled_dot_product_attention(q, k, v, attn_ctx)

        o = self.o_proj(o.reshape(L, -1))
        return o

    @torch.compile
    def qkv_forward(self, x: torch.Tensor):
        if self.use_qkv_proj:
            qkv: torch.Tensor = self.qkv_proj(x)
            q, k, v = qkv.split(
                [
                    self.num_heads * self.head_dim,
                    self.num_query_groups * self.head_dim,
                    self.num_query_groups * self.head_dim,
                ],
                dim=-1,
            )
        else:
            q: torch.Tensor = self.q_proj(x)
            k: torch.Tensor = self.k_proj(x)
            v: torch.Tensor = self.v_proj(x)
        return q, k, v

    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_ctx: AttentionContext,
    ):
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, attn_ctx.slot_mapping)
        if attn_ctx.is_prefill:
            if attn_ctx.block_tables is not None:  # prefix cache
                k, v = k_cache, v_cache
            o = attn_varlen(
                q=q,
                k=k,
                v=v,
                max_seqlen_q=attn_ctx.max_seqlen_q,
                cu_seqlens_q=attn_ctx.cu_seqlens_q,
                max_seqlen_k=attn_ctx.max_seqlen_k,
                cu_seqlens_k=attn_ctx.cu_seqlens_k,
                softmax_scale=self.scale,
                is_causal=True,
            )
        else:  # decode
            o = attn_with_paged_kvcache(
                q=q.unsqueeze(1),
                k_cache=k_cache,
                v_cache=v_cache,
                cache_seqlens=attn_ctx.context_lens,
                block_table=attn_ctx.block_tables,
                softmax_scale=self.scale,
                is_causal=True,
            )
        return o

    def set_cache(
        self,
        max_length: int,
        num_kvcache_blocks: int,
        block_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.block_size = block_size
        self.k_cache = torch.zeros(
            num_kvcache_blocks,
            block_size,
            self.num_query_groups,
            self.head_dim,
            device=device,
            dtype=dtype,
        )
        self.v_cache = torch.zeros(
            num_kvcache_blocks,
            block_size,
            self.num_query_groups,
            self.head_dim,
            device=device,
            dtype=dtype,
        )
        if self.apply_rope:
            self.rope_cos_cache, self.rope_sin_cache = build_rope_cache(
                max_length, self.head_dim, base=self.rope_base, device=device, repeat=False
            )

    def clear_cache(self):
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])
        self.rope_cos_cache = torch.tensor([])
        self.rope_sin_cache = torch.tensor([])

    @property
    def num_kv_heads(self) -> int:
        return self.num_query_groups

    @property
    def kv_head_dim(self) -> int:
        return self.head_dim


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    slot = tl.load(slot_mapping_ptr + idx)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)
