<div align="center">

# OSC-Transformers

**🚀 基于配置文件的模块化 Transformer 模型构建框架**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*灵活、高效、可扩展的 Transformer 模型构建工具*

</div>

## ✨ 特性

- 🔧 **配置驱动**: 通过简单配置文件构建 Transformer 模型
- 🧩 **模块化设计**: 支持自定义注册各类组件
- ⚡ **高性能**: 支持 CUDA Graph 和 Paged Attention

## 🛠️ 支持组件

| 组件类型 | 内置实现 |
|---------|---------|
| 注意力机制 | `PagedAttention` |
| 前馈网络 | `SwiGLU` |
| 归一化 | `RMSNorm` |
| 嵌入层 | `VocabEmbedding` |
| 输出头 | `LMHead` |

## 📦 安装

- 安装 [PyTorch 2.10+](https://pytorch.org/)
- 安装 osc-transformers。注意力 kernel 已由 Triton 实现。
```bash
pip install osc-transformers
```


## 🚀 快速开始


创建 `model.cfg`(Qwen3-0.6B):
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
代码示例：
```python
from osc_transformers import TransformerDecoder, Sequence, SamplingParams

# 构建模型
model = TransformerDecoder.from_config("model.cfg")
model.setup(gpu_memory_utilization=0.9, max_model_len=40960, device="cuda:0")

# 批量推理
seqs = [Sequence(token_ids=[1,2,3,4,5,6,7,8,9,10], sampling_params=SamplingParams(temperature=0.5, max_generate_tokens=1024))]
seqs = model.batch(seqs)

# 流式推理
seq = Sequence(token_ids=[1,2,3,4,5,6,7,8,9,10], sampling_params=SamplingParams(temperature=0.5, max_generate_tokens=1024))
for token in model.stream(seq):
    pass

```

## 📚 推理性能
```bash
osc-transformers bench examples/configs/qwen3-0_6B.cfg --num_seqs 64 --max_input_len 1024 --max_output_len 1024 --gpu_memory_utilization 0.9
```
| 架构 | 模型 |设备 | 吞吐量(tokens/s) |
|---------|---------|---------|---------|
| TransformerDecoder | Qwen3-0.6B | 4090 | 5400 |
| TransformerDecoder | Qwen3-0.6B | 3090 | 4000 |

## 📚 致谢

本项目核心代码主要来自于以下项目：

- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
- [Triton](https://github.com/triton-lang/triton)
- [Liger-Kernel](https://github.com/linkedin/Liger-Kernel)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License
