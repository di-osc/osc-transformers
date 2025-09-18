from osc_transformers import TransformerDecoder
from pathlib import Path
from osc_transformers import TransformerDecoderBuilder
from osc_transformers.normalization import Normalization
from osc_transformers.registry import Registry
import torch

# 使用默认注册的组件构建模型
builder = TransformerDecoderBuilder(num_layers=8)

embedding_config = """
[embedding]
@embedding = VocabEmbedding
num_embeddings = 151936
embedding_dim = 1024
"""
builder.set_embedding(config=embedding_config)

attention_config = """
[attention]
@attention = PagedAttention
in_dim = 1024
num_heads = 16
"""
builder.set_attention(config=attention_config)

head_config = """
[head]
@head = LMHead
in_dim = 1024
out_dim = 151936
bias = True
"""
builder.set_head(config=head_config)

norm_config = """
[normalization]
@normalization = RMSNorm
in_dim = 1024
eps = 1e-5
"""
builder.set_norm(config=norm_config)

feedforward_config = """
[feedforward]
@feedforward = SwiGLU
in_dim = 1024
hidden_dim = 1024
"""
builder.set_feedforward(config=feedforward_config)

model = builder.build()
print(model)


# 自定义Normalization组件构建模型
@Registry.normalization.register("LayerNorm")
class LayerNorm(Normalization):
    def __init__(self, in_dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(in_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.LayerNorm(x, self.weight, self.eps)


norm_config = """
[normalization]
@normalization = LayerNorm
in_dim = 1024
eps = 1e-5
"""
builder.set_norm(config=norm_config)
model = builder.build()
print(model)

# 使用配置文件构建模型
config = Path(__file__).parent / "decoder.cfg"
model = TransformerDecoder.form_config(config=config)
print(model)
