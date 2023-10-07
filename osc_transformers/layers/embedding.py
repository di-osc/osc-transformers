import torch.nn as nn
from ..config import registry
import torch


@registry.layers.register("token_embeddings")
class Embeddings(nn.Module):
    def __init__(self, 
                 n_tokens: int,
                 n_embd: int,):
        super().__init__()
        self.embeddings = nn.Embedding(n_tokens, n_embd)
        
    def forward(self, x):
        return self.embeddings(x)
    
    
    
@registry.layers.register("embeddings.positional")
class EmbeddingsPositional(Embeddings):
    def __init__(self, 
                 n_embeddings: int,
                 embedding_size: int,
                 max_length: int = 512):
        super().__init__(n_embeddings, embedding_size)
        self.positional_embeddings = nn.Embedding(max_length, embedding_size)
        
    def forward(self, x, input_pos=None):
        x = super().forward(x)
        if input_pos is None:
            input_pos = torch.arange(x.shape[1], device=x.device)
        x = x + self.positional_embeddings(input_pos)
        return x