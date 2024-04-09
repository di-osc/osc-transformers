import torch.nn as nn
from osc_transformers.config import registry
from typing import Optional
import torch
from copy import deepcopy



class TransformerEncoderBlock(nn.Module):
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
        **kwargs
    ):
        if self.prenorm:
            x = self.attention(self.attention_norm(x), **kwargs) + x
            x = x + self.feedforward(self.feedforward_norm(x))
        else:
            x = self.attention_norm(self.attention(x, **kwargs) + x)
            x = self.feedforward_norm(self.feedforward(x) + x)
        return x 


@registry.architectures.register("TransformerEncoder")
class TransformerEncoder(nn.Module):
    def __init__(self,
                 n_blocks: int,
                 block_size: int,
                 embedding: nn.Module,
                 attention: nn.Module,
                 feedforward: nn.Module,
                 head: nn.Module,
                 norm: nn.Module,
                 prenorm: bool,
                 ) -> None:
        super().__init__()
        
        self.prenorm = prenorm
        self.n_blocks = n_blocks
        self.embedding = embedding
        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock(attention=deepcopy(attention), attention_norm=deepcopy(norm), feedforward=deepcopy(feedforward), feedforward_norm=deepcopy(norm), prenorm=prenorm) for _ in range(n_blocks)]
        )
        self.head_norm = norm if self.prenorm else None
        self.head = head
        
        self.block_size = block_size
        
    def forward(
        self,
        input_ids: torch.LongTensor,
        input_pos: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        x = self.embedding(input_ids, token_type_ids=token_type_ids, input_pos=input_pos)
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
        if self.prenorm:
            x = self.head(self.head_norm(x))
        else:
            x = self.head(x)
        return x