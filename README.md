<div align="center">

# OSC-Transformers

**🚀 基于配置文件的模块化 Transformer 模型构建框架**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*灵活、高效、可扩展的 Transformer 模型构建工具*

</div>

## ✨ 特性

- 🔧 **配置驱动**: 通过简单配置文件构建 Transformer 模型
- 🧩 **模块化设计**: 支持自定义注册各类组件
- ⚡ **高性能**: 支持 CUDA Graph 和 Paged Attention
- 🎯 **易于使用**: 提供 Builder 模式和配置文件两种方式

## 🛠️ 支持组件

| 组件类型 | 内置实现 |
|---------|---------|
| 注意力机制 | `PagedAttention` |
| 前馈网络 | `SwiGLU` |
| 归一化 | `RMSNorm` |
| 嵌入层 | `VocabEmbedding` |
| 输出头 | `LMHead` |

## 📦 安装

```bash
pip install osc-transformers
```

环境要求：Python >= 3.10, PyTorch >= 2.8.0

## 🚀 快速开始

### 使用配置文件

创建 `model.cfg`:
```toml
[model]
@architecture = "TransformerDecoder"
num_layers = 12
max_length = 2048

[model.attention]
@attention = "PagedAttention"
in_dim = 768
num_heads = 12

[model.embedding]
@embedding = "VocabEmbedding"
num_embeddings = 30000
embedding_dim = 768
```

加载模型：
```python
from osc_transformers import TransformerDecoder
model = TransformerDecoder.from_config("model.cfg")
```


## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License