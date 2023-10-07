from ..config import registry
import torch.nn as nn
from typing import Optional
import torch


class DecoderBlock(nn.Module):
    def __init__(self,
                 attention: nn.Module,
                 attention_norm: nn.Module,
                 mlp: nn.Module,
                 mlp_norm: nn.Module):
        super().__init__()
        self.attention = attention
        self.attention_norm = attention_norm
        self.mlp = mlp
        self.mlp_norm = mlp_norm
        
    def build_kv_cache(self, batch_size: int, max_seq_length: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        self.attention.build_kv_cache(batch_size=batch_size, max_seq_length=max_seq_length, device=device, dtype=dtype)
        
    def clear_kv_cache(self):
        self.attention.kv_cache = None



@registry.layers.register("decoder_block.prenorm")
class PreNormDecoderBlock(DecoderBlock):
        
    def forward(self, 
                x, 
                input_pos: Optional[torch.Tensor] = None, 
                mask: Optional[torch.Tensor] = None, 
                **kwargs):
        x = self.attention(self.attention_norm(x), input_pos=input_pos, mask=mask, **kwargs) + x
        x = x + self.mlp(self.mlp_norm(x))
        return x
    
    
@registry.layers.register("decoder_block.postnorm")
class PostNormDecoderBlock(DecoderBlock):
    """后标准化的解码器块
    """
        
    def forward(self, 
                x, 
                input_pos: Optional[torch.Tensor] = None, 
                mask: Optional[torch.Tensor] = None, 
                **kwargs):
        x = self.attention_norm(x + self.attention(x, input_pos=input_pos, mask=mask, **kwargs))
        x = self.mlp_norm(x + self.mlp(x))
        return x
    
    
class DecoderBlocks(nn.Module):
    def __init__(self, 
                 n_blocks: int, 
                 block: DecoderBlock):
        super().__init__()
        
        
            