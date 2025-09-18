# OSC-Transformers

<div align="center">

**🚀 基于配置文件的模块化 Transformer 模型构建框架**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*灵活、高效、可扩展的 Transformer 模型构建工具*

</div>

## ✨ 主要特性

- 🔧 **配置驱动**: 通过简单的配置文件构建复杂的 Transformer 模型
- 🧩 **模块化设计**: 支持自定义注册各种组件（注意力机制、前馈网络、归一化等）
- ⚡ **高性能优化**: 
  - 支持 CUDA Graph 加速
  - 内置 Paged Attention 机制
  - 高效的内存管理
- 🎯 **易于使用**: 提供多种构建方式，从简单的 API 到复杂的配置文件
- 🔄 **高度可扩展**: 基于注册机制，轻松扩展新的模型组件

## 🛠️ 支持的组件

| 组件类型 | 内置实现 | 描述 |
|---------|---------|------|
| **注意力机制** | `PagedAttention` | 高效的分页注意力实现 |
| **前馈网络** | `SwiGLU` | SwiGLU 激活函数的前馈网络 |
| **归一化** | `RMSNorm` | RMS 归一化层 |
| **嵌入层** | `VocabEmbedding` | 词汇表嵌入层 |
| **输出头** | `LMHead` | 语言模型输出头 |
| **采样器** | `SimpleSampler` | 简单的采样实现 |

## 📦 安装

### 环境要求
- Python >= 3.10
- PyTorch >= 2.8.0

### 安装方式

```bash
pip install osc-transformers --upgrade
```

或从源码安装：

```bash
git clone https://github.com/your-repo/osc-transformers.git
cd osc-transformers
pip install -e .
```

## 🚀 快速开始

### 方式一：使用 Builder 模式

```python
from osc_transformers import TransformerDecoderBuilder

# 创建构建器
builder = TransformerDecoderBuilder(num_layers=8, max_length=1024)

# 配置各个组件
embedding_config = '''
[embedding]
@embedding = VocabEmbedding
num_embeddings = 32000
embedding_dim = 1024
'''
builder.set_embedding(config=embedding_config)

attention_config = '''
[attention]
@attention = PagedAttention
in_dim = 1024
num_heads = 16
'''
builder.set_attention(config=attention_config)

# ... 配置其他组件

# 构建模型
model = builder.build()
```

### 方式二：使用配置文件

创建配置文件 `model.cfg`:

```toml
[model]
@architecture = "TransformerDecoder"
num_layers = 28
max_length = 40960
prenorm = "True"

[model.attention]
@attention = "PagedAttention"
in_dim = 1024
num_heads = 16
head_dim = 128
num_query_groups = 8

[model.embedding]
@embedding = "VocabEmbedding"
num_embeddings = 32000
embedding_dim = 1024

[model.feedforward]
@feedforward = "SwiGLU"
in_dim = 1024
hidden_dim = 3072

[model.head]
@head = "LMHead"
in_dim = 1024
out_dim = 32000

[model.norm]
@normalization = "RMSNorm"
in_dim = 1024
eps = 1e-6
```

加载模型：

```python
from osc_transformers import TransformerDecoder

model = TransformerDecoder.form_config(config="model.cfg")
```

## 🔧 自定义组件

框架支持注册自定义组件，例如自定义归一化层：

```python
from osc_transformers.normalization import Normalization
from osc_transformers.registry import Registry
import torch

@Registry.normalization.register("LayerNorm")
class LayerNorm(Normalization):
    def __init__(self, in_dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(in_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.layer_norm(x, (x.size(-1),), self.weight, eps=self.eps)
```

然后在配置中使用：

```toml
[model.norm]
@normalization = "LayerNorm"
in_dim = 1024
eps = 1e-5
```

## 📚 API 文档

### TransformerDecoder

主要的 Transformer 解码器模型类。

#### 参数

- `num_layers` (int): 解码器层数
- `max_length` (int): 最大序列长度
- `attention` (CausalSelfAttention): 注意力机制
- `embedding` (Embedding): 嵌入层
- `feedforward` (FeedForward): 前馈网络
- `head` (Head): 输出头
- `norm` (Normalization): 归一化层
- `prenorm` (bool): 是否使用预归一化

#### 方法

- `form_config(config, model_section="model", empty_init=True)`: 从配置文件构建模型
- `setup(**kwargs)`: 设置模型（如缓存等）
- `forward(input_ids, attn_ctx)`: 前向传播
- `compute_logits(x)`: 计算输出 logits

### TransformerDecoderBuilder

构建器模式的模型构建类。

#### 方法

- `set_embedding(config, section="embedding")`: 设置嵌入层
- `set_attention(config, section="attention")`: 设置注意力机制
- `set_feedforward(config, section="feedforward")`: 设置前馈网络
- `set_head(config, section="head")`: 设置输出头
- `set_norm(config, section="normalization")`: 设置归一化层
- `build()`: 构建最终模型

## 🎯 使用场景

- **研究原型**: 快速实验不同的 Transformer 架构
- **生产部署**: 高性能的推理服务
- **教学演示**: 理解 Transformer 内部结构
- **模型定制**: 针对特定任务的模型优化

## 🤝 贡献

欢迎贡献代码！请查看我们的贡献指南：

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- 感谢 [Confection](https://github.com/explosion/confection) 提供的配置系统
- 感谢 PyTorch 团队提供的深度学习框架
- 感谢所有贡献者的支持

---

<div align="center">

**如果这个项目对您有帮助，请给我们一个 ⭐️**

</div>