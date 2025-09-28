from pathlib import Path

import torch

from osc_transformers import SamplingParams, Sequence, TransformerDecoder
from osc_transformers.normalization import Normalization
from osc_transformers.registry import Registry

# 使用配置文件构建模型
config = Path(__file__).parent / "configs" / "qwen3-0_6B.cfg"
model = TransformerDecoder.from_config(config=config, empty_init=True)

config = """
[model]
@architecture = "TransformerDecoder"
num_layers = 28
prenorm = "True"

[model.attention]
@attention = "PagedAttention"
in_dim = 1024
num_heads = 16
head_dim = 128
num_query_groups = 8
rope_base = 1000000
q_bias = "False"
k_bias = "False"
v_bias = "False"
o_bias = "False"

[model.attention.k_norm]
@normalization = "RMSNorm"
in_dim = 128
eps = 0.000001

[model.attention.q_norm]
@normalization = "RMSNorm"
in_dim = 128
eps = 0.000001

[model.embedding]
@embedding = "VocabEmbedding"
num_embeddings = 151936
embedding_dim = 1024

[model.feedforward]
@feedforward = "SwiGLU"
in_dim = 1024
hidden_dim = 3072
up_bias = "False"
gate_bias = "False"
down_bias = "False"

[model.head]
@head = "LMHead"
in_dim = 1024
out_dim = 151936
bias = "False"

[model.norm]
@normalization = "RMSNorm"
in_dim = 1024
eps = 0.000001
"""
model = TransformerDecoder.from_config(config=config, empty_init=True)


# 自定义Normalization组件构建模型
@Registry.normalization.register("LayerNorm")
class LayerNorm(Normalization):
    def __init__(self, in_dim: int, eps: float = 1e-5):
        super().__init__()
        self._norm = torch.nn.LayerNorm(in_dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x)


config = """
[model]
@architecture = "TransformerDecoder"
num_layers = 28
prenorm = "True"

[model.attention]
@attention = "PagedAttention"
in_dim = 1024
num_heads = 16
head_dim = 128
num_query_groups = 8
rope_base = 1000000
q_bias = "False"
k_bias = "False"
v_bias = "False"
o_bias = "False"

[model.attention.k_norm]
@normalization = "LayerNorm"
in_dim = 128
eps = 0.000001

[model.attention.q_norm]
@normalization = "LayerNorm"
in_dim = 128
eps = 0.000001

[model.embedding]
@embedding = "VocabEmbedding"
num_embeddings = 151936
embedding_dim = 1024

[model.feedforward]
@feedforward = "SwiGLU"
in_dim = 1024
hidden_dim = 3072
up_bias = "False"
gate_bias = "False"
down_bias = "False"

[model.head]
@head = "LMHead"
in_dim = 1024
out_dim = 151936
bias = "False"

[model.norm]
@normalization = "LayerNorm"
in_dim = 1024
eps = 0.000001
"""

model = TransformerDecoder.from_config(config=config, empty_init=False)


## setup model
model.setup(model_name="qwen3-0_6B")

# batch inference
seqs = [
    Sequence(
        token_ids=[1, 2, 3, 4, 5],
        sampling_params=SamplingParams(temperature=0.5, max_generate_tokens=1024),
    )
]
seqs = model.batch(seqs=seqs)

# stream inference
seq = Sequence(
    token_ids=[1, 2, 3, 4, 5],
    sampling_params=SamplingParams(temperature=0.5, max_generate_tokens=1024),
)
for token in model.stream(seq=seq):
    pass
