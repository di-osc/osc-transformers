from osc_transformers.utils import lazy_load, WeightMap
from osc_transformers import registry, Config, Tokenizer
import json
from pathlib import Path
from lightning import Fabric
import torch
from jsonargparse import CLI 


config_str = """
[model]
@architectures = "TransformerDecoder"
n_blocks = 32
block_size = 4096
prenorm = True

[model.embedding]
@layers = "TokenEmbedding"
n_embeddings = 55296
embedding_size = 4096

[model.attention]
@layers = "CausalSelfAttention"
n_in = 4096
n_heads = 32
n_query_groups = 32
q_bias = false
k_bias = false
v_bias = false
o_bias = false

[model.feedforward]
@layers = "SwiGLU"
n_in = 4096
n_hidden = 11008

[model.head]
@layers = "LMHead"
n_in = 4096
n_out = 55296
bias = false

[model.norm]
@layers = "RMSNorm"
n_in = 4096
eps = 1e-5
"""



def convert_hf_checkpoint(checkpoint_dir: str, save_path: str, n_blocks: int = 32):
    
    checkpoint_dir = Path(checkpoint_dir)
    
    with open(checkpoint_dir / 'pytorch_model.bin.index.json', 'r') as f:
        index = json.load(f)
        
    files = [checkpoint_dir / file  for file in set(index['weight_map'].values())]
    
    sd = {}
    wmap = WeightMap.get_llama2_weight_map(num_blocks=n_blocks)
    for file in files:
        weights = torch.load(file)
        for key in weights:
            if key not in wmap:
                continue
            sd[wmap[key]] = weights[key]
    
    fabric = Fabric(devices=1, accelerator='gpu', precision='16-true')
    with fabric.init_module(empty_init=True):
        config = Config().from_str(config_str)
        model = registry.resolve(config)['model']
    model.load_state_dict(state_dict=sd, strict=True)
    
    tokenizer = Tokenizer(checkpoint_dir=checkpoint_dir)
    
    state = {'model': model, 'tokenizer': tokenizer, 'config': config}
    
    fabric.save(state=state, path=save_path)
    
if __name__ == "__main__":
    CLI(convert_hf_checkpoint)