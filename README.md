<div align="center">

# OSC-Transformers

**🚀 Configuration-driven Modular Transformer Model Building Framework**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Flexible, efficient, and extensible Transformer model building tools*

**中文文档**: [README-zh.md](README-zh.md)

</div>

## ✨ Features

- 🔧 **Configuration Driven**: Build Transformer models through simple configuration files
- 🧩 **Modular Design**: Support custom registration of various components
- ⚡ **High Performance**: Support CUDA Graph and Paged Attention

## 🛠️ Supported Components

| Component Type | Built-in Implementation |
|---------|---------|
| Attention Mechanism | `PagedAttention` |
| Feedforward Network | `SwiGLU` |
| Normalization | `RMSNorm` |
| Embedding Layer | `VocabEmbedding` |
| Output Head | `LMHead` |

## 📦 Installation

- Install [latest version PyTorch](https://pytorch.org/)
- Install [flash-attn](https://github.com/Dao-AILab/flash-attention): It is recommended to download the official pre-built whl package to avoid compilation issues
- Install osc-transformers
```bash
pip install osc-transformers
```

## 🚀 Quick Start


Create `model.cfg`(Qwen3-0.6B):
```toml
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
```
Code example:
```python
from osc_transformers import TransformerDecoder, Sequence, SamplingParams

# Build model
model = TransformerDecoder.from_config("model.cfg")
model.setup(gpu_memory_utilization=0.9, max_model_len=40960, device="cuda:0")

# Batch inference
seqs = [Sequence(token_ids=[1,2,3,4,5,6,7,8,9,10], sampling_params=SamplingParams(temperature=0.5, max_generate_tokens=1024))]
seqs = model.batch(seqs)

# Streaming inference
seq = Sequence(token_ids=[1,2,3,4,5,6,7,8,9,10], sampling_params=SamplingParams(temperature=0.5, max_generate_tokens=1024))
for token in model.stream(seq):
    pass

```

## 📚 Inference Performance
```bash
osc-transformers bench examples/decoder.cfg --num_seqs 64 --max_input_len 1024 --max_output_len 1024 --gpu_memory_utilization 0.9
```
| Device | Throughput(tokens/s) |
|---------|---------|
| 4090 | 5400 |

## 📚 Acknowledgments

The core code of this project mainly comes from the following projects:

- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)
- [Liger-Kernel](https://github.com/linkedin/Liger-Kernel)

## 🤝 Contributing

Welcome to submit Issue and Pull Request!

## 📄 License

MIT License
