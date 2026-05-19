import argparse
import math
import sys
import time
from pathlib import Path

import torch
from torch.nn.attention.varlen import varlen_attn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from osc_transformers.ops.attention import attn_varlen


def _time_cuda(fn, warmup: int, repeat: int) -> tuple[torch.Tensor, float]:
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        out = fn()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000 / repeat
    return out, elapsed_ms


def _report(actual: torch.Tensor, expected: torch.Tensor, rtol: float, atol: float) -> None:
    diff = (actual.float() - expected.float()).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    p99_err = torch.quantile(diff.flatten(), 0.99).item()
    print(f"abs_error: max={max_err:.8g} p99={p99_err:.8g} mean={mean_err:.8g}")
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    print(f"parity: PASS rtol={rtol} atol={atol}")


def bench_varlen(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    q_lens = [args.seqlen_q - 7, args.seqlen_q, args.seqlen_q + 5]
    k_lens = q_lens if args.causal else [args.seqlen_k - 11, args.seqlen_k, args.seqlen_k + 3]
    cu_q = torch.tensor([0, *torch.cumsum(torch.tensor(q_lens), dim=0).tolist()], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, *torch.cumsum(torch.tensor(k_lens), dim=0).tolist()], dtype=torch.int32, device="cuda")
    q = torch.randn(cu_q[-1].item(), args.q_heads, args.head_dim, device="cuda", dtype=dtype)
    k = torch.randn(cu_k[-1].item(), args.kv_heads, args.head_dim, device="cuda", dtype=dtype)
    v = torch.randn(cu_k[-1].item(), args.kv_heads, args.head_dim, device="cuda", dtype=dtype)
    scale = 1.0 / math.sqrt(args.head_dim)

    triton_fn = lambda: attn_varlen(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(k_lens),
        softmax_scale=scale,
        is_causal=args.causal,
    )
    baseline_fn = lambda: varlen_attn(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max(q_lens),
        max(k_lens),
        is_causal=args.causal,
    )

    triton_out, triton_ms = _time_cuda(triton_fn, args.warmup, args.repeat)
    torch_out, torch_ms = _time_cuda(baseline_fn, args.warmup, args.repeat)
    _report(triton_out, torch_out, args.rtol, args.atol)
    print(f"latency_ms: triton={triton_ms:.3f} torch_varlen={torch_ms:.3f} ratio={triton_ms / torch_ms:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Triton varlen attention with torch.nn.attention.varlen.")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--seqlen-q", type=int, default=1024)
    parser.add_argument("--seqlen-k", type=int, default=1024)
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.q_heads % args.kv_heads != 0:
        raise ValueError("--q-heads must be divisible by --kv-heads")
    bench_varlen(args)


if __name__ == "__main__":
    main()
