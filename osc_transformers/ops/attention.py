import torch
import triton
import triton.language as tl
from torch.nn.attention.varlen import varlen_attn


@triton.jit
def _paged_kvcache_decode_forward_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    o_ptr,
    cache_seqlens_ptr,
    block_table_ptr,
    stride_q_b: tl.constexpr,
    stride_q_s: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_k_block: tl.constexpr,
    stride_k_token: tl.constexpr,
    stride_k_h: tl.constexpr,
    stride_k_d: tl.constexpr,
    stride_v_block: tl.constexpr,
    stride_v_token: tl.constexpr,
    stride_v_h: tl.constexpr,
    stride_v_d: tl.constexpr,
    stride_o_b: tl.constexpr,
    stride_o_s: tl.constexpr,
    stride_o_h: tl.constexpr,
    stride_o_d: tl.constexpr,
    stride_block_table_b: tl.constexpr,
    stride_block_table_page: tl.constexpr,
    scale: tl.constexpr,
    q_head_group_size: tl.constexpr,
    MAX_CACHE_LEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Grid layout:
    #   axis 0: batch sequence id
    #   axis 1: KV head id
    #   axis 2: a chunk of query heads that share the same KV head
    #
    # Decode attention has one query token per sequence. For GQA/MQA, multiple
    # query heads attend to the same KV head, so each program computes a small
    # query-head tile for one (batch, kv_head) pair.
    pid_b = tl.program_id(0)
    pid_kv_h = tl.program_id(1)
    pid_group_h = tl.program_id(2)

    cache_len = tl.load(cache_seqlens_ptr + pid_b)
    # q_heads are the logical query-head ids handled by this program. Some
    # entries are padding when q_head_group_size < BLOCK_H; valid_h prevents
    # those lanes from loading/writing.
    offs_h = pid_group_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)
    q_heads = pid_kv_h * q_head_group_size + offs_h
    valid_h = offs_h < q_head_group_size

    # q shape is (batch, 1, n_q_heads, head_dim). The sequence dimension is 1
    # for decode, so stride_q_s is intentionally unused.
    q = tl.load(
        q_ptr + pid_b * stride_q_b + q_heads[:, None] * stride_q_h + offs_d[None, :] * stride_q_d,
        mask=valid_h[:, None] & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )

    m_i = tl.full((BLOCK_H,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_H,), tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), tl.float32)

    # Walk the paged KV cache in token order. block_table maps logical page
    # indices in this sequence to physical cache block ids:
    #   token offset -> page offset + token offset within page -> physical block
    # Invalid pages are masked before load; safe_block_ids only keeps pointer
    # arithmetic well-defined for masked lanes.
    for start_n in range(0, MAX_CACHE_LEN, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        page_offsets = offs_n // BLOCK_SIZE
        token_offsets = offs_n - page_offsets * BLOCK_SIZE
        block_ids = tl.load(
            block_table_ptr + pid_b * stride_block_table_b + page_offsets * stride_block_table_page,
            mask=offs_n < cache_len,
            other=0,
        )
        valid = offs_n < cache_len
        safe_block_ids = tl.maximum(block_ids, 0)

        # k_cache/v_cache shape is
        #   (num_blocks, block_size, n_kv_heads, head_dim)
        # BLOCK_D is the next power-of-two for vectorized dot/load. The
        # HEAD_DIM mask drops padded columns for non power-of-two head dims.
        k = tl.load(
            k_cache_ptr
            + safe_block_ids[:, None] * stride_k_block
            + token_offsets[:, None] * stride_k_token
            + pid_kv_h * stride_k_h
            + offs_d[None, :] * stride_k_d,
            mask=valid[:, None] & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        )
        v = tl.load(
            v_cache_ptr
            + safe_block_ids[:, None] * stride_v_block
            + token_offsets[:, None] * stride_v_token
            + pid_kv_h * stride_v_h
            + offs_d[None, :] * stride_v_d,
            mask=valid[:, None] & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        )

        # Online softmax. m_i and l_i carry the running row max and normalizer,
        # avoiding materializing all attention logits for long cache lengths.
        scores = tl.dot(q, tl.trans(k)) * scale
        scores = tl.where(valid[None, :] & valid_h[:, None], scores, -float("inf"))
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        p = tl.exp(scores - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    # Normalize accumulated weighted values and write only real query-head lanes.
    acc = acc / l_i[:, None]
    tl.store(
        o_ptr + pid_b * stride_o_b + q_heads[:, None] * stride_o_h + offs_d[None, :] * stride_o_d,
        acc,
        mask=valid_h[:, None] & (offs_d[None, :] < HEAD_DIM),
    )


def attn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float | None = None,
    is_causal: bool = True,
) -> torch.Tensor:
    assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3, "q, k, v must be 3D packed tensors"
    assert q.is_cuda and k.is_cuda and v.is_cuda, "PyTorch varlen attention requires CUDA tensors"
    assert cu_seqlens_q.is_cuda and cu_seqlens_k.is_cuda, "cu_seqlens tensors must be on CUDA"
    assert q.shape[-1] == k.shape[-1] == v.shape[-1], "q, k, v head dimensions must match"
    assert k.shape[1] == v.shape[1], "k and v must have the same number of heads"
    assert q.shape[1] % k.shape[1] == 0, "q heads must be divisible by kv heads"

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    cu_seqlens_q = cu_seqlens_q.contiguous()
    cu_seqlens_k = cu_seqlens_k.contiguous()

    _, n_q_heads, head_dim = q.shape
    softmax_scale = softmax_scale or head_dim**-0.5
    default_scale = head_dim**-0.5
    if softmax_scale != default_scale:
        q = q * (softmax_scale / default_scale)

    return varlen_attn(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        is_causal=is_causal,
    )


def attn_with_paged_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor,
    softmax_scale: float | None = None,
    is_causal: bool = True,
) -> torch.Tensor:
    assert q.dim() == 4 and q.shape[1] == 1, "decode attention expects q shape (batch, 1, nheads, head_dim)"
    assert k_cache.dim() == 4 and v_cache.dim() == 4, "k_cache and v_cache must be 4D paged tensors"
    assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda, "Triton attention requires CUDA tensors"
    assert cache_seqlens.is_cuda and block_table.is_cuda, "cache metadata tensors must be on CUDA"
    assert q.shape[-1] == k_cache.shape[-1] == v_cache.shape[-1], "q, k, v head dimensions must match"
    assert k_cache.shape[:3] == v_cache.shape[:3], "k_cache and v_cache shapes must match"
    assert q.shape[2] % k_cache.shape[2] == 0, "q heads must be divisible by kv heads"

    q = q.contiguous()
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()
    cache_seqlens = cache_seqlens.contiguous()
    block_table = block_table.contiguous()

    batch, _, n_q_heads, head_dim = q.shape
    assert head_dim <= 256, "Triton paged KV cache attention only supports head_dim <= 256"
    assert cache_seqlens.numel() == batch, "cache_seqlens length must match batch size"
    assert block_table.shape[0] >= batch, "block_table batch dimension must cover q batch size"

    softmax_scale = softmax_scale or head_dim**-0.5
    output = torch.empty_like(q)
    if batch == 0:
        return output

    block_size = k_cache.shape[1]
    max_cache_len = block_table.shape[1] * block_size
    q_head_group_size = n_q_heads // k_cache.shape[2]
    # Triton 3.x requires dot operands to have M/N/K >= 16. GQA/MQA often has
    # fewer query heads per KV head, so compute a masked 16-head tile even when
    # only 1 or 2 lanes are real.
    block_h = 16
    # Larger BLOCK_N reduces loop count but increases shared memory pressure.
    # 64 is safe for head_dim <= 128 on RTX 3060; >128 head dims use 32.
    block_n = 32 if head_dim > 128 else 64
    block_d = triton.next_power_of_2(head_dim)
    num_warps = 8 if head_dim >= 128 else 4
    # Launch one program per batch, KV head, and query-head group chunk.
    grid = (batch, k_cache.shape[2], triton.cdiv(q_head_group_size, block_h))

    with torch.cuda.device(q.device.index):
        _paged_kvcache_decode_forward_kernel[grid](
            q,
            k_cache,
            v_cache,
            output,
            cache_seqlens,
            block_table,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            k_cache.stride(3),
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            v_cache.stride(3),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            output.stride(3),
            block_table.stride(0),
            block_table.stride(1),
            float(softmax_scale),
            q_head_group_size,
            max_cache_len,
            block_size,
            head_dim,
            BLOCK_H=block_h,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            num_warps=num_warps,
        )
    return output
