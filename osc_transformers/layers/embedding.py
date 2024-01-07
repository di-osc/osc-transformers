import torch.nn as nn
from ..config import registry


@registry.layers.register("TokenEmbedding")
class TokenEmbedding(nn.Module):
    def __init__(self, 
                 n_embeddings: int,
                 embedding_size: int):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=n_embeddings, embedding_dim=embedding_size)
        
    def forward(self, x):
        return self.embed(x)