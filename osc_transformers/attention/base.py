import torch.nn as nn


class CausalSelfAttention(nn.Module):
    def set_cache(self, **kwargs):
        """Set all caches for the attention layer, including kv cache, rope cache, etc.

        Raises:
            NotImplementedError: This method should be implemented by the subclass.
        """
        raise NotImplementedError
