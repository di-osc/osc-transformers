from typing import Mapping, List, Any, Self, TYPE_CHECKING
from copy import deepcopy
from pathlib import Path

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

if TYPE_CHECKING:
    pass


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


@Registry.architecture.register("TransformerDecoder")
class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        max_length: int,
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

        self.max_length = max_length

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

        if self.max_length < L:
            raise ValueError(
                f"Cannot forward sequence of length {L}, max seq length is only {self.max_length}."
            )

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

    def setup(self, **kwargs) -> None:
        for layer in self.layers:
            layer.attention.set_cache(**kwargs)

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
    def __init__(self, num_layers: int, max_length: int, prenorm: bool = True):
        super().__init__()
        self.num_layers = num_layers
        self.max_length = max_length
        self.prenorm = prenorm
        self.embedding = None
        self.head = None
        self.norm = None
        self.attention = None
        self.feedforward = None

    def set_embedding(self, config: Config | str, section: str = "embedding") -> Self:
        with torch.device("meta"):
            self.embedding = self.resolve_module(config, section)

    def set_head(self, config: Config | str, section: str = "head") -> Self:
        with torch.device("meta"):
            self.head = self.resolve_module(config, section)

    def set_norm(self, config: Config | str, section: str = "normalization") -> Self:
        with torch.device("meta"):
            self.norm = self.resolve_module(config, section)

    def set_attention(self, config: Config | str, section: str = "attention") -> Self:
        with torch.device("meta"):
            self.attention = self.resolve_module(config, section)

    def set_feedforward(
        self, config: Config | str, section: str = "feedforward"
    ) -> Self:
        with torch.device("meta"):
            self.feedforward = self.resolve_module(config, section)

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
        model = TransformerDecoder(
            num_layers=self.num_layers,
            max_length=self.max_length,
            prenorm=self.prenorm,
            embedding=self.embedding,
            attention=self.attention,
            feedforward=self.feedforward,
            head=self.head,
            norm=self.norm,
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
