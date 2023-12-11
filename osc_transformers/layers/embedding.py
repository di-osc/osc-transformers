import torch.nn as nn
from ..config import registry
import torch


@registry.layers.register("TokenEmbeddings")
class TokenEmbeddings(nn.Module):
    def __init__(self, 
                 n_tokens: int,
                 n_embd: int,):
        super().__init__()
        self.token_embeddings = nn.Embedding(n_tokens, n_embd)
        self.position
    def forward(self, x):
        return self.embeddings(x)