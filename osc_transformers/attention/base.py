from dataclasses import dataclass

import torch.nn as nn
import torch


class CausalSelfAttention(nn.Module):
    def set_cache(
        self,
        max_length: int,
        num_kvcache_blocks: int,
        block_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        """Set all caches for the attention layer, including kv cache, rope cache, etc.

        Raises:
            NotImplementedError: This method should be implemented by the subclass.
        """
        raise NotImplementedError

    def clear_cache(self):
        raise NotImplementedError

    @property
    def num_kv_heads(self) -> int:
        raise NotImplementedError

    @property
    def kv_head_dim(self) -> int:
        raise NotImplementedError


@dataclass
class AttentionContext:
    is_prefill: bool = False
    input_pos: torch.Tensor | None = None
    # varlen attention
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    # paged attention
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

    def reset_run_info(self):
        self.input_pos = None
        self.is_prefill = False
        self.cu_seqlens_k = None
        self.cu_seqlens_q = None
        self.max_seqlen_k = 0
        self.max_seqlen_q = 0
        self.slot_mapping = None
        self.context_lens = None
        self.block_tables = None
