<div align="center">

# OSC-Transformers

**🚀 Configuration-driven Modular Transformer Model Building Framework**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Flexible, efficient, and extensible Transformer model building tools*

[📖 Documentation](https://github.com/di-osc/osc-transformers) | [🚀 Quick Start](#-quick-start) | [🤝 Contributing](#-contributing)

**中文文档**: [README-zh.md](README-zh.md)

</div>

## ✨ Features

- 🔧 **Configuration Driven**: Build Transformer models through simple configuration files
- 🧩 **Modular Design**: Support custom registration of various components
- ⚡ **High Performance**: Support CUDA Graph and Paged Attention optimization
- 🎯 **Easy to Use**: Provide configuration file and programmatic ways to build models
- 🔄 **Flexible Inference**: Support batch generation and streaming generation
- 📦 **Plug and Play**: Rich built-in components, ready to use
- 🏗️ **Type Safe**: Complete type annotations, support IDE intelligent prompts

## 🏗️ Project Architecture

```
osc_transformers/
├── attention/          # Attention mechanism components
├── embedding/          # Embedding layer components
├── feedforward/        # Feedforward network components
├── head/              # Output head components
├── normalization/     # Normalization components
├── sampler/           # Sampler components
├── block_manager.py   # Block manager
├── decoder.py         # Transformer decoder
├── registry.py        # Component registry system
├── scheduler.py       # Scheduler
└── sequence.py        # Sequence management
```

## 🛠️ Supported Components

| Component Type | Built-in Implementation | Description |
|---------|---------|------|
| **Attention Mechanism** | `PagedAttention` | Efficient Paged Attention implementation |
| **Feedforward Network** | `SwiGLU` | SwiGLU activation function feedforward network |
| **Normalization** | `RMSNorm` | RMS normalization |
| **Embedding Layer** | `VocabEmbedding` | Vocabulary embedding layer |
| **Output Head** | `LMHead` | Language model output head |
| **Sampler** | `SimpleSampler` | Simple greedy sampler |

## 📦 Installation

- Install [latest version PyTorch](https://pytorch.org/)
- Install [flash-attn](https://github.com/Dao-AILab/flash-attention): It is recommended to download the official pre-built whl package to avoid compilation issues
- Install osc-transformers
```bash
pip install osc-transformers
```

## 🚀 Quick Start

### Method 1: Using Configuration File

1. Create configuration file `model.cfg`:

```toml
[model]
@architecture = "TransformerDecoder"
num_layers = 28
prenorm = true

[model.attention]
@attention = "PagedAttention"
in_dim = 1024
num_heads = 16
head_dim = 128
num_query_groups = 8
rope_base = 1000000
q_bias = false
k_bias = false
v_bias = false
o_bias = false

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
up_bias = false
gate_bias = false
down_bias = false

[model.head]
@head = "LMHead"
in_dim = 1024
out_dim = 151936
bias = false

[model.norm]
@normalization = "RMSNorm"
in_dim = 1024
eps = 0.000001
```

2. Load and use the model:

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

## 🔧 Custom Components

### Registering Custom Components

```python
import torch.nn as nn
from osc_transformers.registry import Registry
from osc_transformers.normalization import Normalization

@Registry.normalization.register("CustomRMSNorm")
class CustomRMSNorm(Normalization):
    def __init__(self, in_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(in_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
```

### Using in Configuration

```toml
[model.norm]
@normalization = "CustomRMSNorm"
in_dim = 768
eps = 1e-6
```

## 📚 API Documentation

### Core Classes

- **`TransformerDecoder`**: Transformer decoder main class
- **`Sequence`**: Sequence management class
- **`Registry`**: Component registry system

### Component Interfaces

- **`CausalSelfAttention`**: Self-attention mechanism interface
- **`FeedForward`**: Feedforward network interface
- **`Normalization`**: Normalization interface
- **`Embedding`**: Embedding layer interface
- **`Head`**: Output head interface
- **`Sampler`**: Sampler interface

## 🧪 Testing and Examples

View `examples/` directory for complete examples:

- `examples/decoder.cfg`: Complete model configuration file
- `examples/build_decoder.py`: Builder pattern example
- `test.ipynb`: Jupyter test notebook

Run tests:

```bash
python -m pytest tests/
```

## 📊 Inference Performance

```bash
osc-transformers bench examples/decoder.cfg --num_seqs 64 --max_input_len 1024 --max_output_len 1024 --gpu_memory_utilization 0.9
```

| Device | Throughput |
|---------|---------|
| RTX 4090 | 5200 tokens/s |

## 🤝 Contributing

Welcome to contribute code! Please follow these steps:

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Create a Pull Request


## 🙏 Acknowledgments

The core code of this project mainly comes from the following projects:

- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)

## 📞 Contact

- Author: wangmengdi
- Email: 790990241@qq.com
- Project Homepage: https://github.com/di-osc/osc-transformers

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
