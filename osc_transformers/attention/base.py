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

    @property
    def num_kv_heads(self) -> int:
        raise NotImplementedError

    @property
    def kv_head_dim(self) -> int:
        raise NotImplementedError
