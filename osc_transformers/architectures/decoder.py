from ..config import registry
import torch.nn as nn
from typing import Optional, Tuple
import torch

RoPECache = Tuple[torch.Tensor, torch.Tensor]


class DecoderBlock(nn.Module):
    def __init__(self,
                 attention: nn.Module,
                 attention_norm: nn.Module,
                 mlp: nn.Module,
                 mlp_norm: nn.Module,
                 pre_norm: bool = True):
        super().__init__()
        self.attention = attention
        self.attention_norm = attention_norm
        self.mlp = mlp
        self.mlp_norm = mlp_norm
        self.pre_norm = pre_norm
        
    def build_kv_cache(self, batch_size: int, max_seq_length: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        self.attention.build_kv_cache(batch_size=batch_size, max_seq_length=max_seq_length, device=device, dtype=dtype)
        
    def clear_kv_cache(self):
        self.attention.kv_cache = None
        
    def forward(
        self,
        x,
        input_pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        if self.pre_norm:
            x = self.attention(self.attention_norm(x), input_pos=input_pos, mask=mask, **kwargs) + x
            x = x + self.mlp(self.mlp_norm(x))
        else:
            x = self.attention_norm(self.attention(x, input_pos=input_pos, mask=mask, **kwargs)) + x
            x = self.mlp_norm(self.mlp(x)) + x
        return x 


@registry.architectures.register("Decoder.v1")
class Decoder(nn.Module):
    def __init__(self, 
                 n_token_embeddings: int,
                 embedding_size: int,
                 n_blocks: int,
                 block_size: int,
                 attention: nn.Module,
                 mlp: nn.Module,
                 norm: nn.Module) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(n_token_embeddings, embedding_dim=embedding_size)
        self.blocks = nn.ModuleList([DecoderBlock(attention=attention, attention_norm=norm, mlp=mlp, mlp_norm=norm) for _ in range(n_blocks)])
        self.head = nn.Linear(embedding_size, n_token_embeddings)
        
        self.block_size = block_size
        self.max_seq_length = block_size
        self.mask_cache : Optional[torch.Tensor] = None
        
    @property
    def max_seq_length(self):
        return self._max_seq_length
    
    @max_seq_length.setter
    def max_seq_length(self, value):
        if value > self.block_size:
            raise ValueError("max_seq_length must be less than or equal to block_size")
        self._max_seq_length = value
        
    def build_caches(self, 
                       batch_size: int, 
                       max_seq_length: int, 
                       device: Optional[torch.device] = None, 
                       dtype: Optional[torch.dtype] = None):
        for block in self.blocks:
            block: DecoderBlock
            block.build_kv_cache(batch_size=batch_size, 
                                 max_seq_length=max_seq_length, 
                                 device=device, 
                                 dtype=dtype)
        self.mask_cache = torch.tril(torch.ones((max_seq_length, max_seq_length), device=device, dtype=torch.bool))
            
    def clear_caches(self):
        for block in self.blocks:
            block: DecoderBlock
            block.clear_kv_cache()
        self.mask_cache = None


@registry.layers.register("decoder.roformer")
class RoformerDecoder(Decoder):
        
    def forward(self, x, input_pos: Optional[torch.Tensor] = None):
        T = x.size(1)
        if self.max_seq_length < T:
            raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")

        if input_pos is not None:  # use the kv cache
            cos = self.cos.index_select(0, input_pos)
            sin = self.sin.index_select(0, input_pos)
            if self.mask_cache is None:
                raise TypeError("You need to call `model.build_kv_cache()`")
            mask = self.mask_cache.index_select(2, input_pos)
        else:
            cos = self.cos[:T]
            sin = self.sin[:T]
            mask = None
        x = self.token_embeddings(x)
        for block in self.blocks:
            x = block(x, input_pos=input_pos, cos=cos, sin=sin, mask=mask)
        x = self.head(x)
        return x
    
    @property
    def max_seq_length(self):
        return self._max_seq_length
    
    @max_seq_length.setter
    def max_seq_length(self, value):
        if value > self.block_size:
            raise ValueError("max_seq_length must be less than or equal to block_size")
        self._max_seq_length = value
        if not hasattr(self, "sin") or not hasattr(self, "cos"):
            cos, sin = self.build_rope_cache()
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        elif value != self.cos.shape[0]:
            cos, sin = self.build_rope_cache(device=self.cos.device)
            self.cos = cos
            self.sin = sin
        
    def build_rope_cache(self, device: Optional[torch.device] = None):
        head_size = self.blocks[0].attention.head_size
        cos, sin = build_rope_cache(seq_len=self.max_seq_length, 
                                    n_elem=head_size, 
                                    dtype=torch.get_default_dtype(), 
                                    device=device)
        return cos, sin
            
        
        
    
def build_rope_cache(seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000, condense_ratio: int = 1) -> RoPECache:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device) / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)

    cos, sin = torch.cos(idx_theta), torch.sin(idx_theta)

    # this is to mimic the behaviour of complex32, else we will get different results
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        return cos.half(), sin.half()
    return cos, sin