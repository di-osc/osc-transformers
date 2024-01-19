from ..config import registry, Config

LLAMA_2_BASE_CONFIG = """
[model]
@architectures = "TransformerDecoder"
n_blocks = {n_blocks}
block_size = {block_size}
prenorm = True

[model.embedding]
@layers = "TokenEmbedding"
n_embeddings = {n_embeddings}
embedding_size = {embedding_size}

[model.attention]
@layers = "CausalSelfAttention"
n_in = {embedding_size}
n_heads = {n_heads}
n_query_groups = {n_query_groups}
q_bias = false
k_bias = false
v_bias = false
o_bias = false

[model.feedforward]
@layers = "SwiGLU"
n_in = {embedding_size}
n_hidden = 11008

[model.head]
@layers = "LMHead"
n_in = {embedding_size}
n_out = {n_embeddings}
bias = false

[model.norm]
@layers = "RMSNorm"
n_in = {embedding_size}
eps = 1e-5
"""

@registry.configs.register("chinese-alpaca-2-7B")
def chinese_alpaca_2_7b_config():
    CHINESE_ALPACA_2_7B_CONFIG = LLAMA_2_BASE_CONFIG.format(
        n_blocks=32, 
        block_size=4096, 
        n_embeddings=55296, 
        embedding_size=4096, 
        n_heads=32, 
        n_query_groups=32
    )
    return Config().from_str(CHINESE_ALPACA_2_7B_CONFIG)

@registry.configs.register("chinese-alpaca-2-1.3B")
def chinese_alpaca_2_1_3b_config():
    CHINESE_ALPACA_2_1_3B_CONFIG = LLAMA_2_BASE_CONFIG.format(
        n_blocks=4,
        block_size=4096,
        n_embeddings=55296,
        embedding_size=4096,
        n_heads=32,
        n_query_groups=32
    )
    return Config().from_str(CHINESE_ALPACA_2_1_3B_CONFIG)