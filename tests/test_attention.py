import math

import pytest
import torch
from torch.nn.attention.varlen import varlen_attn

from osc_transformers.ops.attention import attn_varlen, attn_with_paged_kvcache


def _torch_attn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float,
    is_causal: bool,
) -> torch.Tensor:
    outputs = []
    cu_q = cu_seqlens_q.cpu().tolist()
    cu_k = cu_seqlens_k.cpu().tolist()
    q_head_group_size = q.shape[1] // k.shape[1]

    for i in range(len(cu_q) - 1):
        qi = q[cu_q[i] : cu_q[i + 1]]
        ki = k[cu_k[i] : cu_k[i + 1]]
        vi = v[cu_k[i] : cu_k[i + 1]]
        seq_outputs = []

        for h in range(q.shape[1]):
            kv_h = h // q_head_group_size
            scores = torch.matmul(qi[:, h], ki[:, kv_h].transpose(0, 1)) * softmax_scale
            if is_causal:
                q_len, k_len = scores.shape
                q_pos = torch.arange(q_len, device=q.device)[:, None]
                k_pos = torch.arange(k_len, device=q.device)[None, :]
                scores = scores.masked_fill(k_pos > q_pos + k_len - q_len, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            seq_outputs.append(torch.matmul(probs, vi[:, kv_h]))
        outputs.append(torch.stack(seq_outputs, dim=1))

    return torch.cat(outputs, dim=0)


def _torch_paged_kvcache_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    outputs = []
    block_size = k_cache.shape[1]
    q_head_group_size = q.shape[2] // k_cache.shape[2]

    for b in range(q.shape[0]):
        seq_len = cache_seqlens[b].item()
        block_ids = block_table[b, : (seq_len + block_size - 1) // block_size]
        k_parts = []
        v_parts = []
        for i, block_id in enumerate(block_ids.tolist()):
            start = i * block_size
            end = min(seq_len - start, block_size)
            k_parts.append(k_cache[block_id, :end])
            v_parts.append(v_cache[block_id, :end])
        kb = torch.cat(k_parts, dim=0)
        vb = torch.cat(v_parts, dim=0)

        head_outputs = []
        for h in range(q.shape[2]):
            kv_h = h // q_head_group_size
            scores = torch.matmul(q[b, 0, h], kb[:, kv_h].transpose(0, 1)) * softmax_scale
            probs = torch.softmax(scores, dim=-1)
            head_outputs.append(torch.matmul(probs, vb[:, kv_h]))
        outputs.append(torch.stack(head_outputs, dim=0))

    return torch.stack(outputs, dim=0).unsqueeze(1)


@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("n_q_heads,n_kv_heads", [(4, 4), (8, 2)])
@pytest.mark.parametrize("head_dim", [32, 64, 128])
def test_triton_attn_varlen_matches_torch(is_causal, n_q_heads, n_kv_heads, head_dim):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triton attention")

    torch.manual_seed(0)
    dtype = torch.float16
    q_lens = [3, 17, 29]
    k_lens = q_lens if is_causal else [5, 19, 31]
    cu_q = torch.tensor([0, *torch.cumsum(torch.tensor(q_lens), dim=0).tolist()], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, *torch.cumsum(torch.tensor(k_lens), dim=0).tolist()], dtype=torch.int32, device="cuda")

    q = torch.randn(cu_q[-1].item(), n_q_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(cu_k[-1].item(), n_kv_heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(cu_k[-1].item(), n_kv_heads, head_dim, device="cuda", dtype=dtype)
    scale = 1.0 / math.sqrt(head_dim)

    actual = attn_varlen(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(k_lens),
        softmax_scale=scale,
        is_causal=is_causal,
    )
    expected = _torch_attn_varlen(q.float(), k.float(), v.float(), cu_q, cu_k, scale, is_causal).to(dtype)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_attn_varlen_matches_torch_varlen(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triton attention")

    torch.manual_seed(0)
    n_q_heads, n_kv_heads, head_dim = 16, 8, 128
    q_lens = [1017, 1024, 1029]
    cu_q = torch.tensor([0, *torch.cumsum(torch.tensor(q_lens), dim=0).tolist()], dtype=torch.int32, device="cuda")
    cu_k = cu_q.clone()
    q = torch.randn(cu_q[-1].item(), n_q_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(cu_q[-1].item(), n_kv_heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(cu_q[-1].item(), n_kv_heads, head_dim, device="cuda", dtype=dtype)
    scale = 1.0 / math.sqrt(head_dim)

    actual = attn_varlen(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(q_lens),
        softmax_scale=scale,
        is_causal=True,
    )
    expected = varlen_attn(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max(q_lens),
        max(q_lens),
        is_causal=True,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("n_q_heads,n_kv_heads", [(4, 4), (8, 2), (8, 1), (16, 1)])
@pytest.mark.parametrize("head_dim", [32, 64, 128])
def test_triton_paged_kvcache_decode_matches_torch(n_q_heads, n_kv_heads, head_dim):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triton attention")

    torch.manual_seed(0)
    dtype = torch.float16
    batch = 5
    block_size = 16
    cache_lens = [1, 7, 16, 23, 41]
    max_blocks_per_seq = (max(cache_lens) + block_size - 1) // block_size
    total_blocks = batch * max_blocks_per_seq

    k_cache = torch.randn(total_blocks, block_size, n_kv_heads, head_dim, device="cuda", dtype=dtype)
    v_cache = torch.randn(total_blocks, block_size, n_kv_heads, head_dim, device="cuda", dtype=dtype)
    q = torch.randn(batch, 1, n_q_heads, head_dim, device="cuda", dtype=dtype)
    cache_seqlens = torch.tensor(cache_lens, dtype=torch.int32, device="cuda")
    block_table = torch.full((batch, max_blocks_per_seq), -1, dtype=torch.int32, device="cuda")

    next_block = 0
    for b, seq_len in enumerate(cache_lens):
        num_blocks = (seq_len + block_size - 1) // block_size
        block_table[b, :num_blocks] = torch.arange(next_block, next_block + num_blocks, device="cuda")
        next_block += num_blocks

    scale = 1.0 / math.sqrt(head_dim)
    actual = attn_with_paged_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        softmax_scale=scale,
        is_causal=True,
    )
    expected = _torch_paged_kvcache_decode(
        q.float(), k_cache.float(), v_cache.float(), cache_seqlens, block_table, scale
    )
    expected = expected.to(dtype)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
