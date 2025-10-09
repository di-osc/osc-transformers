import torch
from loguru import logger

from ..registry import Registry
from .base import Sampler


@Registry.sampler.register("SimpleSampler")
class SimpleSampler(Sampler):
    def __init__(self, top_k: int = 100, top_p: float = 0.7):
        super().__init__()
        self.top_k = top_k
        self.top_p = top_p
        logger.info("🎯 Using Simple Sampler: TopK → Temperature → Softmax → Sample")

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # topk sampling
        value, indices = torch.topk(logits, min(self.top_k, logits.size(-1)))
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, indices, value)
        # temperature sampling
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        # sample
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens

    # @torch.compile(fullgraph=True, dynamic=True)
    # def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
    #     # topk sampling
    #     value, indices = torch.topk(logits, min(self.top_k, logits.size(-1)))
    #     logits = torch.full_like(logits, float("-inf")).scatter_(-1, indices, value)
    #     # topp sampling
    #     sorted_logits, sorted_indices = torch.sort(logits, descending=False)
    #     cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    #     # Example:
    #     # sorted_probs=[0.1, 0.15, 0.2, 0.25, 0.3] -> sorted_cumprobs=[0.1, 0.25, 0.45, 0.7, 1.0]
    #     # sorted_indices_to_remove = [1, 1, 0, 0, 0] if top_p=0.7
    #     sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
    #     # Keep at least 1 token always to prevent the case where no token is selected
    #     # In this case the most probable one is always kept
    #     sorted_indices_to_remove[..., -1:] = 0
    #     indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
    #     logits = logits.masked_fill(indices_to_remove, float("-inf"))
    #     # temperature sampling
    #     logits = logits.float().div_(temperatures.unsqueeze(dim=1))
    #     # sample
    #     probs = torch.softmax(logits, dim=-1)
    #     sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
    #     return sample_tokens
