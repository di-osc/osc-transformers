import pytest
import torch

from osc_transformers.architectures.autoregressive import AutoRegressiveTransformer
from osc_transformers.attention.paged_attention import PagedAttention
from osc_transformers.embedding.vocab import VocabEmbedding
from osc_transformers.feedforward.swiglu import SwiGLU
from osc_transformers.head.lm import LMHead
from osc_transformers.normalization.rmsnorm import TorchRMSNorm
from osc_transformers.sampler import SamplingParams
from osc_transformers.sequence import Sequence


def _make_zero_layer_model() -> AutoRegressiveTransformer:
    hidden_size = 8
    return AutoRegressiveTransformer(
        num_layers=0,
        attention=PagedAttention(in_dim=hidden_size, num_heads=2, head_dim=4),
        embedding=VocabEmbedding(num_embeddings=16, embedding_dim=hidden_size),
        feedforward=SwiGLU(in_dim=hidden_size, hidden_dim=16),
        head=LMHead(in_dim=hidden_size, out_dim=16, bias=False),
        norm=TorchRMSNorm(in_dim=hidden_size),
    ).eval()


def test_forward_accepts_inputs_embeds_equivalent_to_token_embeddings():
    model = _make_zero_layer_model()
    input_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    inputs_embeds = model.embedding(input_ids)

    with torch.no_grad():
        token_path = model(input_ids=input_ids, attn_ctx=None)
        embed_path = model(inputs_embeds=inputs_embeds, attn_ctx=None)

    torch.testing.assert_close(embed_path, token_path)


def test_forward_rejects_both_input_ids_and_inputs_embeds():
    model = _make_zero_layer_model()
    input_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    inputs_embeds = model.embedding(input_ids)

    with pytest.raises(ValueError, match="exactly one"):
        model(input_ids=input_ids, inputs_embeds=inputs_embeds, attn_ctx=None)


def test_sequence_accepts_prompt_embeds_matching_prompt_length():
    token_ids = [1, 2, 3]
    prompt_embeds = torch.randn(3, 8)

    seq = Sequence(token_ids=token_ids, sampling_params=SamplingParams(), prompt_embeds=prompt_embeds)

    assert seq.prompt_embeds is prompt_embeds
    assert len(seq) == 3


def test_sequence_rejects_prompt_embeds_with_wrong_length():
    with pytest.raises(ValueError, match="prompt_embeds"):
        Sequence(token_ids=[1, 2, 3], sampling_params=SamplingParams(), prompt_embeds=torch.randn(2, 8))
