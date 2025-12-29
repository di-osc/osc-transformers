import time
from collections.abc import Generator, Mapping
from copy import deepcopy
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from typing import Any

import torch
import torch.nn as nn
from confection import Config
from loguru import logger

from ..attention import AttentionContext, CausalSelfAttention
from ..embedding import Embedding
from ..feedforward import FeedForward
from ..head import Head
from ..normalization import Normalization
from ..registry import Registry
from ..sampler import Sampler, SimpleSampler
from ..scheduler import Scheduler
from ..sequence import Sequence


@Registry.architecture.register("Autoregressive")
class AutoregressiveTransformer(nn.Module):
    def __init__(
        self,
        num_layers: int,
        attention: CausalSelfAttention,
        embedding: Embedding,
        feedforward: FeedForward,
        head: Head,
        norm: Normalization,
        prenorm: bool = True,
    ):
        super().__init__()

        self.prenorm = prenorm
        self.num_layers = num_layers
        self.embedding = embedding
        self.layers: list[TransformerLayer] = nn.ModuleList(
            [
                TransformerLayer(
                    attention=deepcopy(attention),
                    attention_norm=deepcopy(norm),
                    feedforward=deepcopy(feedforward),
                    feedforward_norm=deepcopy(norm),
                    prenorm=prenorm,
                    layer_id=i,
                )
                for i in range(num_layers)
            ]
        )
        self.head_norm = deepcopy(norm) if self.prenorm else None
        self.head = deepcopy(head)
        self.sampler: Sampler = None

        self.enable_cuda_graph = False
        self.scheduler: Scheduler = None

        self.stop_event = Event()
        self.name = "AutoregressiveTransformer"
        self.run_thread = None

        self.dtype = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attn_ctx: AttentionContext,
    ):
        """Forward pass of the AutoregressiveTransformer.

        Args:
            input_ids (torch.Tensor): Input token ids. shape = (seq_length)
            attn_ctx (AttentionContext): Attention context.
        """
        assert len(input_ids.shape) == 1, "input must be 1d"
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x, attn_ctx=attn_ctx)
        if self.prenorm:
            x = self.head_norm(x)
        return x

    def compute_logits(self, x: torch.Tensor, attn_ctx: AttentionContext) -> torch.Tensor:
        if attn_ctx.is_prefill:
            last_indices = attn_ctx.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logis = self.head(x)
        return logis

    def _run_loop(self):
        while not self.stop_event.is_set():
            try:
                scheduled_seqs, is_prefill = self.scheduler.schedule()
                if len(scheduled_seqs) == 0:
                    time.sleep(0.01)
                    continue
                if is_prefill:
                    scheduled_seqs = self.prefill(scheduled_seqs)
                else:
                    scheduled_seqs = self.decode(scheduled_seqs)
                self.scheduler.check_finished(scheduled_seqs)
            except Exception as e:
                logger.error(e)
                self.stop_event.set()
                self.scheduler.set_all_failed(str(e))
                break
        logger.info("🛑 inference loop stopped, you can call setup() again to start a new loop")

    def prepare_prefill(self, seqs: list[Sequence]) -> tuple[torch.Tensor, AttentionContext]:
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens :])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.scheduler.block_manager.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.scheduler.block_manager.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:  # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        attn_ctx = AttentionContext(
            input_pos=positions,
            is_prefill=True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
        )
        return input_ids, attn_ctx

    @torch.inference_mode()
    def prefill(self, seqs: list[Sequence]) -> list[Sequence]:
        if not seqs:
            return seqs
        input_ids, attn_ctx = self.prepare_prefill(seqs)
        logits = self.compute_logits(self.forward(input_ids, attn_ctx), attn_ctx)
        temperatures = self.prepare_sample([seq for seq in seqs])
        token_ids = self.sampler(logits, temperatures).tolist()
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
        attn_ctx.reset_run_info()
        return seqs

    def prepare_decode(self, seqs: list[Sequence]) -> tuple[torch.Tensor, AttentionContext]:
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token_id)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(
                seq.block_table[-1] * self.scheduler.block_manager.block_size + seq.last_block_num_tokens - 1
            )
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        attn_ctx = AttentionContext(
            input_pos=positions,
            is_prefill=False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        return input_ids, attn_ctx

    @torch.inference_mode()
    def decode(self, seqs: list[Sequence]) -> list[Sequence]:
        if not seqs:
            return seqs
        input_ids, attn_ctx = self.prepare_decode(seqs)
        if not self.enable_cuda_graph:
            logits = self.compute_logits(self.forward(input_ids, attn_ctx), attn_ctx)
        else:
            bs = input_ids.size(0)
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["input_pos"][:bs] = attn_ctx.input_pos
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = attn_ctx.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = attn_ctx.context_lens
            graph_vars["block_tables"][:bs, : attn_ctx.block_tables.size(1)] = attn_ctx.block_tables
            graph.replay()
            logits = self.compute_logits(graph_vars["outputs"][:bs], attn_ctx)
        temperatures = self.prepare_sample(seqs=seqs)
        token_ids = self.sampler(logits, temperatures).tolist()
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
        attn_ctx.reset_run_info()
        return seqs

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def setup(
        self,
        max_model_len: int = 4096,
        gpu_memory_utilization: float | None = None,
        eos: int | list[int] | None = None,
        max_num_seqs: int = 512,
        block_size: int = 256,
        cuda_graph: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        model_name: str = "AutoregressiveTransformer",
        sampler: Sampler = None,
        start_run_thread: bool = True,
    ) -> None:
        default_dtype = torch.get_default_dtype()
        self.dtype = dtype
        if self.run_thread is not None:
            logger.info(f"🔄 Re-initializing {model_name} with device: {device} and dtype: {dtype}")
            self.stop_event.set()
            self.run_thread.join()
            self.run_thread = None
            self.stop_event = Event()
            self.clear_cache()
            torch.cuda.reset_peak_memory_stats(device=device)
        else:
            logger.info(f"🔄 Initializing {model_name} with device: {device} and dtype: {dtype}")
        self.name = model_name
        torch.set_default_device(device)
        torch.set_default_dtype(dtype)
        self.to(device=device, dtype=dtype)
        # Record model memory after moving to GPU
        torch.cuda.synchronize()
        model_memory = self.model_size(include_embeddings=True)
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = self.layers[0].attention.num_kv_heads
        kv_head_dim = self.layers[0].attention.kv_head_dim
        block_bytes = 2 * self.num_layers * block_size * num_kv_heads * kv_head_dim * dtype.itemsize
        if gpu_memory_utilization is not None:
            num_kvcache_blocks = int(total * gpu_memory_utilization - used - peak + current) // block_bytes
        else:
            num_kvcache_blocks = max_model_len // block_size
        if num_kvcache_blocks <= 0:
            logger.error(
                "❌ Not enough GPU memory to allocate KV cache, current: {format_bytes(current)}, peak: {format_bytes(peak)}, free: {format_bytes(free)}, total: {format_bytes(total)}"
            )
            return
        max_num_batched_tokens = num_kvcache_blocks * block_size
        kv_cache_memory = num_kvcache_blocks * block_bytes

        total_memory_usage = model_memory + kv_cache_memory
        memory_utilization = total_memory_usage / total * 100
        logger.info(
            f"💾 GPU Memory allocated: {format_bytes(total_memory_usage)} "
            f"({memory_utilization:.1f}% of {format_bytes(total)}) | "
            f"Model: {format_bytes(model_memory)} | KV Cache: {format_bytes(kv_cache_memory)} | "
            f"Max length: {max_model_len:,} tokens | KV Cache capacity: {max_num_batched_tokens:,} tokens"
        )
        self.scheduler = Scheduler(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            eos=eos,
            num_kvcache_blocks=num_kvcache_blocks,
            kvcache_block_size=block_size,
        )
        for layer in self.layers:
            layer.attention.set_cache(
                num_kvcache_blocks=num_kvcache_blocks,
                block_size=block_size,
                max_length=max_model_len,
                device=device,
                dtype=dtype,
            )
        if sampler is None:
            sampler = SimpleSampler()
        self.sampler = sampler
        if cuda_graph:
            logger.info(f"⚡ Capturing CUDA Graph for acceleration ({max_num_seqs} sequences)")
            self.enable_cuda_graph = True
            self.capture_cudagraph(
                max_num_seqs=max_num_seqs,
                max_model_len=max_model_len,
                block_size=block_size,
            )
        if start_run_thread:
            logger.info("🚀 Starting inference loop in background thread")
            self.run_thread = Thread(target=self._run_loop, daemon=True)
            self.run_thread.name = model_name
            self.run_thread.start()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

    def warmup_model(self, num_seqs: int, seq_len: int):
        """Warmup the model by prefilling the cache for a given number of sequences and sequence length"""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        self.prefill(seqs)
        torch.cuda.empty_cache()

    def clear_cache(self):
        for layer in self.layers:
            layer.attention.clear_cache()
        # Clear CUDA graphs if they exist
        if hasattr(self, "graphs"):
            self.graphs = {}
        if hasattr(self, "graph_pool"):
            self.graph_pool = None
        if hasattr(self, "graph_vars"):
            self.graph_vars = {}
        torch.cuda.empty_cache()

    def batch(self, seqs: list[Sequence], timeout: float | None = None):
        assert self.run_thread is not None, f"{self.name} is not running, please call setup() first"
        response_queue = Queue()
        num_seqs = len(seqs)
        results = []
        for seq in seqs:
            self.scheduler.add(seq, response_queue)
        while not self.stop_event.is_set():
            seq = response_queue.get(timeout=timeout)
            results.append(seq)
            if len(results) == num_seqs:
                break
        return results

    def stream(self, seq: Sequence, timeout: float | None = None) -> Generator[int, None, None]:
        assert self.run_thread is not None, f"{self.name} is not running, please call setup() first"
        response_queue = Queue()
        seq.stream_response = True
        self.scheduler.add(seq, response_queue)
        while not self.stop_event.is_set():
            token_id = response_queue.get(timeout=timeout)
            if token_id == seq.end_char:
                break
            yield token_id

    @torch.inference_mode()
    def capture_cudagraph(self, max_num_seqs: int, max_model_len: int, block_size: int):
        max_bs = min(max_num_seqs, 512)
        max_num_blocks = (max_model_len + block_size - 1) // block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        input_pos = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, self.head.in_dim)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            attn_ctx = AttentionContext(
                input_pos=input_pos[:bs],
                is_prefill=False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )
            outputs[:bs] = self.forward(input_ids[:bs], attn_ctx)  # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.forward(input_ids[:bs], attn_ctx)  # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            attn_ctx.reset_run_info()

        self.graph_vars = dict(
            input_ids=input_ids,
            input_pos=input_pos,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = True):
        # 保证在用torch.device('meta')构建模型后, 可以运行model.to('cuda:xxx'),不然会由于cos和sin是meta data而报错
        return super().load_state_dict(state_dict, strict, assign)

    def model_size(self, include_embeddings: bool = True) -> int:
        """Calculate the model size.

        Args:
            include_embeddings (bool, optional): Include embeddings in the model size. Defaults to True.

        Returns:
            int: Model size in MB
        """
        import itertools

        model_size = 0
        for n, children in self.named_children():
            if n == "embedding" and not include_embeddings:
                continue
            model_size += sum(
                [p.numel() * p.dtype.itemsize for p in itertools.chain(children.parameters(), children.buffers())]
            )
        return model_size

    @classmethod
    def from_config(
        cls,
        config: Config | str | Path,
        model_section: str = "model",
        empty_init: bool = False,
    ) -> "AutoregressiveTransformer":
        if isinstance(config, Path):
            config = Config().from_disk(config)
        elif isinstance(config, str):
            try:
                config = Config().from_disk(config)
            except Exception:
                config = Config().from_str(config)

        if model_section not in config:
            raise ValueError(f"{model_section} section is required")
        if empty_init:
            with torch.device("meta"):
                model: AutoregressiveTransformer = Registry.resolve(config=config)[model_section]
        else:
            model: AutoregressiveTransformer = Registry.resolve(config=config)[model_section]
        return model.eval()


class TransformerLayer(nn.Module):
    def __init__(
        self,
        attention: CausalSelfAttention,
        attention_norm: Normalization,
        feedforward: FeedForward,
        feedforward_norm: Normalization,
        prenorm: bool = True,
        layer_id: int = 0,
    ):
        super().__init__()
        self.attention = attention
        self.attention_norm = attention_norm
        self.feedforward = feedforward
        self.feedforward_norm = feedforward_norm
        self.prenorm = prenorm
        self.layer_id = layer_id

    def forward(
        self,
        x,
        attn_ctx: AttentionContext,
    ):
        if self.prenorm:
            x = self.attention(self.attention_norm(x), attn_ctx=attn_ctx) + x
            x = self.feedforward(self.feedforward_norm(x)) + x
        else:
            raise NotImplementedError("Only prenorm is supported")
        return x


def format_bytes(bytes_val: int) -> str:
    """Format bytes to human readable format"""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f} TB"
