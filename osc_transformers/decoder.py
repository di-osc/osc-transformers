from typing import Mapping, List, Any, Tuple, Generator
from copy import deepcopy
from pathlib import Path
from queue import Queue
import time
from threading import Thread

import torch
import torch.nn as nn
from confection import Config
from wasabi import msg

from .attention import AttentionContext, CausalSelfAttention
from .registry import Registry
from .embedding import Embedding
from .feedforward import FeedForward
from .head import Head
from .normalization import Normalization
from .sampler import Sampler, SimpleSampler
from .sequence import Sequence
from .scheduler import Scheduler


@Registry.architecture.register("TransformerDecoder")
class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        attention: CausalSelfAttention,
        embedding: Embedding,
        feedforward: FeedForward,
        head: Head,
        norm: Normalization,
        prenorm: bool = True,
        sampler: Sampler = None,
    ):
        super().__init__()

        self.prenorm = prenorm
        self.num_layers = num_layers
        self.embedding = embedding
        self.layers: List[TransformerDecoderLayer] = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    attention=deepcopy(attention),
                    attention_norm=deepcopy(norm),
                    feedforward=deepcopy(feedforward),
                    feedforward_norm=deepcopy(norm),
                    prenorm=prenorm,
                )
                for _ in range(num_layers)
            ]
        )
        self.head_norm = norm if self.prenorm else None
        self.head = head
        self.sampler = sampler or SimpleSampler()

        self.enable_cuda_graph = False
        self.scheduler = None

        self.input_queue = Queue()

    def forward(
        self,
        input_ids: torch.Tensor,
        attn_ctx: AttentionContext,
    ):
        """Forward pass of the TransformerDecoder.

        Args:
            input_ids (torch.Tensor): Input token ids. shape = (seq_length)
            attn_ctx (AttentionContext): Attention context.
        """
        assert len(input_ids.shape) == 1, "input must be 1d"
        L = input_ids.size()[0]
        if attn_ctx.input_pos is None:
            attn_ctx.input_pos = torch.arange(L, dtype=torch.int32)

        x = self.embedding(input_ids)

        for layer in self.layers:
            x = layer(x, attn_ctx=attn_ctx)

        if self.prenorm:
            x = self.head_norm(x)

        if attn_ctx.is_prefill:
            last_indices = attn_ctx.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()

        return x

    def compute_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

    def main_loop(self):
        while True:
            scheduled_seqs, is_prefill = self.scheduler.schedule()
            if len(scheduled_seqs) == 0:
                time.sleep(0.01)
                continue
            if is_prefill:
                self.prefill(scheduled_seqs)
            else:
                self.decode(scheduled_seqs)

    def prefill(self, seqs: List[Tuple[Sequence, Queue]]) -> List[Sequence]:
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq, _ in seqs:
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
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:  # prefix cache
            block_tables = self.prepare_block_tables([seq[0] for seq in seqs])
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        cu_seqlens_q = torch.tensor(
            cu_seqlens_q, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(
            cu_seqlens_k, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
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
        logits = self.compute_logits(self.forward(input_ids, attn_ctx))
        temperatures = self.prepare_sample([seq[0] for seq in seqs])
        token_ids = self.sampler(logits, temperatures).tolist()
        seqs = self.scheduler.postprocess(seqs, token_ids)
        return seqs

    def decode(self, seqs: list[Tuple[Sequence, Queue]]) -> list[Sequence]:
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq, _ in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq))
            context_lens.append(len(seq))
            slot_mapping.append(
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        context_lens = torch.tensor(
            context_lens, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables([seq[0] for seq in seqs])
        attn_ctx = AttentionContext(
            input_pos=positions,
            is_prefill=False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        if not self.enable_cuda_graph:
            logits = self.compute_logits(self.forward(input_ids, attn_ctx))
        else:
            bs = input_ids.size(0)
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            for k, v in graph_vars.items():
                if k != "outputs":
                    v.zero_()
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["input_pos"][:bs] = attn_ctx.input_pos
            graph_vars["slot_mapping"][:bs] = attn_ctx.slot_mapping
            graph_vars["context_lens"][:bs] = attn_ctx.context_lens
            graph_vars["block_tables"][:bs, : attn_ctx.block_tables.size(1)] = (
                attn_ctx.block_tables
            )
            graph.replay()
            logits = self.compute_logits(graph_vars["outputs"][:bs])
        temperatures = self.prepare_sample([seq[0] for seq in seqs])
        token_ids = self.sampler(logits, temperatures).tolist()
        seqs = self.scheduler.postprocess(seqs, token_ids)
        return seqs

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(
            temperatures, dtype=torch.float32, pin_memory=True
        ).cuda(non_blocking=True)
        return temperatures

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs
        ]
        block_tables = torch.tensor(
            block_tables, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        return block_tables

    def setup(
        self,
        max_num_seqs: int,
        max_model_len: int,
        num_kvcache_blocks: int,
        max_num_batched_tokens: int | None = None,
        eos: int | None = None,
        block_size: int = 128,
        cuda_graph: bool = False,
    ) -> None:
        if max_num_batched_tokens is None:
            max_num_batched_tokens = max_model_len * 4
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
            )
        if cuda_graph:
            self.capture_cudagraph(
                max_num_seqs=max_num_seqs, max_model_len=max_model_len
            )
        self.run_thread = Thread(target=self.main_loop, daemon=True)
        self.run_thread.start()

    def batch(self, seqs: list[Sequence]):
        response_queue = Queue()
        num_seqs = len(seqs)
        results = []
        for seq in seqs:
            self.scheduler.add(seq, response_queue)
        for seq in response_queue.get():
            results.append(seq)
            if len(results) == num_seqs:
                break
        return results

    def stream(self, seq: Sequence) -> Generator[int, None, None]:
        response_queue = Queue()
        self.scheduler.add(seq, response_queue)
        for token_id in response_queue.get():
            yield token_id
            if token_id == seq.end_char:
                break

    @torch.inference_mode()
    def capture_cudagraph(self, max_num_seqs: int, max_model_len: int):
        max_bs = min(max_num_seqs, 512)
        max_num_blocks = (max_model_len + self.block_size - 1) // self.block_size
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
            outputs[:bs] = self.model(input_ids[:bs], attn_ctx)  # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], attn_ctx)  # capture
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

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = True
    ):
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
                [
                    p.numel() * p.dtype.itemsize
                    for p in itertools.chain(children.parameters(), children.buffers())
                ]
            )
        return model_size / 1024 / 1024

    @classmethod
    def form_config(
        cls,
        config: Config | str | Path,
        model_section: str = "model",
        empty_init: bool = True,
    ) -> "TransformerDecoder":
        if isinstance(config, Path):
            config = Config().from_disk(config)
        elif isinstance(config, str):
            if Path(config).exists():
                config = Config().from_disk(config)
            else:
                config = Config().from_str(config)
        if model_section not in config:
            msg.fail(f"{model_section} section is required")
        if empty_init:
            with torch.device("meta"):
                model = Registry.resolve(config=config)[model_section]
        else:
            model = Registry.resolve(config=config)[model_section]
        return model


class TransformerDecoderBuilder:
    def __init__(self, num_layers: int, prenorm: bool = True):
        super().__init__()
        self.num_layers = num_layers
        self.prenorm = prenorm
        self.embedding = None
        self.head = None
        self.norm = None
        self.attention = None
        self.feedforward = None
        self.sampler = None

    def set_embedding(
        self, config: Config | str, section: str = "embedding"
    ) -> Embedding:
        with torch.device("meta"):
            self.embedding = self.resolve_module(config, section)
        return self.embedding

    def set_head(self, config: Config | str, section: str = "head") -> Head:
        with torch.device("meta"):
            self.head = self.resolve_module(config, section)
        return self.head

    def set_norm(
        self, config: Config | str, section: str = "normalization"
    ) -> Normalization:
        with torch.device("meta"):
            self.norm = self.resolve_module(config, section)
        return self.norm

    def set_attention(
        self, config: Config | str, section: str = "attention"
    ) -> CausalSelfAttention:
        with torch.device("meta"):
            self.attention = self.resolve_module(config, section)
        return self.attention

    def set_feedforward(
        self, config: Config | str, section: str = "feedforward"
    ) -> FeedForward:
        with torch.device("meta"):
            self.feedforward = self.resolve_module(config, section)
        return self.feedforward

    def set_sampler(self, config: Config | str, section: str = "sampler") -> Sampler:
        with torch.device("meta"):
            self.sampler = self.resolve_module(config, section)
        return self.sampler

    def build(self) -> "TransformerDecoder":
        if self.embedding is None:
            msg.fail("embedding is required")
        if self.head is None:
            msg.fail("head is required")
        if self.norm is None:
            msg.fail("norm is required")
        if self.attention is None:
            msg.fail("attention is required")
        if self.feedforward is None:
            msg.fail("feedforward is required")
        if self.sampler is None:
            msg.fail("sampler is required")
        model = TransformerDecoder(
            num_layers=self.num_layers,
            prenorm=self.prenorm,
            embedding=self.embedding,
            attention=self.attention,
            feedforward=self.feedforward,
            head=self.head,
            norm=self.norm,
            sampler=self.sampler,
        )
        return model

    def resolve_module(self, config: Config | str, section: str) -> nn.Module:
        if isinstance(config, str):
            config = Config().from_str(config)
        if section not in config:
            msg.fail(f"{section} section is required")
        with torch.device("meta"):
            model = Registry.resolve(config=config)[section]
        return model


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        attention: CausalSelfAttention,
        attention_norm: Normalization,
        feedforward: FeedForward,
        feedforward_norm: Normalization,
        prenorm: bool = True,
    ):
        super().__init__()
        self.attention = attention
        self.attention_norm = attention_norm
        self.feedforward = feedforward
        self.feedforward_norm = feedforward_norm
        self.prenorm = prenorm

    def forward(
        self,
        x,
        attn_ctx: AttentionContext,
    ):
        if self.prenorm:
            x = (
                self.attention(
                    self.attention_norm(x),
                    attn_ctx=attn_ctx,
                )
                + x
            )
            x = x + self.feedforward(self.feedforward_norm(x))
        else:
            x = self.attention_norm(
                self.attention(
                    x,
                    attn_ctx=attn_ctx,
                )
                + x
            )
            x = self.feedforward_norm(self.feedforward(x) + x)
        return x
