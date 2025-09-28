import torch
import torch.nn as nn

from ..registry import Registry
from .base import Embedding


@Registry.embedding.register("VocabEmbedding")
class VocabEmbedding(Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

    def forward(self, x: torch.Tensor, **kwargs):
        return self.embed(x)
