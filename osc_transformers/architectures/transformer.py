from ..config import registry
from ..layers.attention import KVCache
import torch.nn as nn
from typing import Optional, Tuple, List
import torch
from copy import deepcopy


RoPECache = Tuple[torch.Tensor, torch.Tensor]


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        attention: nn.Module,
        attention_norm: nn.Module,
        feedforward: nn.Module,
        feedforward_norm: nn.Module,
        prenorm: bool = True
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
        input_pos: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ):
        if self.prenorm:
            x = self.attention(self.attention_norm(x), input_pos=input_pos, attention_mask=attention_mask, cos=cos, sin=sin) + x
            x = x + self.feedforward(self.feedforward_norm(x))
        else:
            x = self.attention_norm(self.attention(x, input_pos=input_pos, attention_mask=attention_mask, sin=sin, cos=cos)) + x
            x = self.feedforward_norm(self.feedforward(x)) + x
        return x 
    
    
@registry.architectures.register("TransformerDecoder")
class TransformerDecoder(nn.Module):
    def __init__(self, 
                 n_blocks: int,
                 block_size: int,
                 embedding: nn.Module,
                 attention: nn.Module,
                 feedforward: nn.Module,
                 head: nn.Module,
                 norm: nn.Module,
                 prenorm: bool,
                 kv_cache: Optional[KVCache] = None):
        super().__init__()
        
        self.prenorm = prenorm
        self.n_blocks = n_blocks
        self.embedding = embedding
        self.blocks = nn.ModuleList(
            [TransformerDecoderBlock(attention=deepcopy(attention), attention_norm=deepcopy(norm), feedforward=deepcopy(feedforward), feedforward_norm=deepcopy(norm), prenorm=prenorm) for _ in range(n_blocks)]
        )
        self.head_norm = norm if self.prenorm else None
        self.head = head
        
        self.block_size = block_size
        self.max_seq_length = block_size
        
        self.mask_cache: Optional[torch.Tensor] = None
        
        if kv_cache:
            self.kv_caches = [deepcopy(kv_cache()) for _ in range(n_blocks)]
        
    @property
    def kv_caches(self) -> List[KVCache]:
        return [block.attention.kv_cache for block in self.blocks]
    
    @kv_caches.setter
    def kv_caches(self, value: List[KVCache]):
        assert len(value) == len(self.blocks), "Number of kv_caches must match number of blocks"
        for block, kv_cache in zip(self.blocks, value):
            block.attention.kv_cache = kv_cache
        
    def reset_parameters(self) -> None:
        # Trigger resetting the rope-cache
        self.max_seq_length = self.block_size
        
    def setup_caches(self, 
                     batch_size: int, 
                     max_length: Optional[int] = None, 
                     kv_cache: Optional[KVCache] = None,
                     device: Optional[torch.device] = None, 
                     dtype: Optional[torch.dtype] = None):
        if kv_cache:
            assert isinstance(kv_cache, KVCache), "kv_cache must be an instance of KVCache"
            self.kv_caches = [deepcopy(kv_cache) for _ in range(self.n_blocks)]
        if not max_length:
            max_length = self.block_size
        else:
            assert max_length <= self.block_size, "max_length must be less than or equal to block_size"
        self.setup_rope_cache(max_length=max_length, device=device)
        self.setup_kv_caches(batch_size, max_length, device, dtype)
    
    def setup_kv_caches(self, batch_size: int, max_length: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        """当前decoder前向推理的时候，需要将encoder的key和value缓存起来，以便decoder的每个block都可以重复使用这些key和value。
        mask cache仅在decode阶段使用, 所以跟kv_cache一起设置。

        Args:
            batch_size (int): 最大批次推理的大小
            max_length (int): 最大序列长度
            device (Optional[torch.device], optional): 推理设备. Defaults to None.
            dtype (Optional[torch.dtype], optional): 推理数据类型. Defaults to None.
        """
        for block in self.blocks:
            block.attention.set_kv_cache(batch_size=batch_size, max_seq_length=max_length, device=device, dtype=dtype)
        
        self.mask_cache = torch.tril(torch.ones((max_length, max_length), device=device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)

    def setup_rope_cache(self, max_length: int, device: Optional[torch.device] = None) -> None:
        head_size = self.blocks[0].attention.head_size
        cos, sin = build_rope_cache(seq_len=max_length, 
                                    n_elem=head_size, 
                                    dtype=torch.get_default_dtype(), 
                                    device=device)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        
    def forward(self, input_ids: torch.Tensor, input_pos: Optional[torch.Tensor] = None):
        """Forward pass of the TransformerDecoder.

        Args:
            input_ids (torch.Tensor): Input token ids. shape = (batch_size, seq_length)
            input_pos (Optional[torch.Tensor], optional): Input position ids. prefill stage shape = (batch_size, seq_length) decode stage shape = (batch_size, 1). Defaults to None.
        """
        
        B, L = input_ids.size()
        
        if self.max_seq_length < L:
            raise ValueError(f"Cannot forward sequence of length {L}, max seq length is only {self.max_seq_length}.")

        if input_pos is not None:
            # use rope cache
            cos = self.cos.index_select(0, input_pos)
            sin = self.sin.index_select(0, input_pos)
            
            if self.mask_cache is None:
                raise TypeError("You need to call `model.build_kv_cache()`")
            mask = self.mask_cache.index_select(2, input_pos)
        else:
            cos = self.cos[:L]
            sin = self.sin[:L]
            mask = None
            
        x = self.embedding(input_ids)
        
        for block in self.blocks:
            x = block(x, input_pos=input_pos, cos=cos, sin=sin, attention_mask=mask)
            
        if self.prenorm:
            x = self.head_norm(x)
            
        x = self.head(x)
        
        return x
            
    
def build_rope_cache(seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000, condense_ratio: int = 1) -> RoPECache:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)

    cos, sin = torch.cos(idx_theta), torch.sin(idx_theta)

    return cos, sin