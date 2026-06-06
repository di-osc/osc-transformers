import torch

from osc_transformers.sampler.simple import SimpleSampler


def test_simple_sampler_uses_greedy_argmax_when_temperature_is_zero():
    sampler = SimpleSampler()
    logits = torch.tensor([[0.1, 2.0, 1.0], [4.0, 0.0, 3.0]])
    temperatures = torch.tensor([0.0, 0.0])

    token_ids = sampler(logits, temperatures)

    assert token_ids.tolist() == [1, 0]
