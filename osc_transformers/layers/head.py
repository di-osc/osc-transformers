import torch.nn as nn
from ..config import registry



@registry.layers.register("head.one")
class OneHead(nn.Module):
    def __init__(self, 
                 embedding_size: int, 
                 vocab_size: int):
        super().__init__()
        self.one = nn.Linear(embedding_size, vocab_size)
        
    def forward(self, x):
        return self.one(x)