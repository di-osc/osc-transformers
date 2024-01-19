from osc_transformers.utils import WeightMap
from osc_transformers import registry, Config, Tokenizer
import json
from pathlib import Path
from lightning import Fabric
from lightning.fabric.utilities.load import _lazy_load as lazy_load
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



def convert_hf_checkpoint(checkpoint_dir: str, save_path: str):
    
    checkpoint_dir: Path = Path(checkpoint_dir)
    index_file = checkpoint_dir / 'pytorch_model.bin.index.json'
    if index_file.exists():
        with open(index_file, 'r') as f:
            index = json.load(f)
        
        files = [checkpoint_dir / file  for file in set(index['weight_map'].values())]
    else:
        files = [checkpoint_dir / 'pytorch_model.bin']
    
    with open(checkpoint_dir / 'config.json', 'r') as f:
        hf_config = json.load(f)
        
    config = Config().from_str(config_str)
    config['model']['n_blocks'] = hf_config['num_hidden_layers']
    config['model']['block_size'] = hf_config['max_length']
    config['model']['embedding']['n_embeddings'] = hf_config['vocab_size']
    config['model']['embedding']['embedding_size'] = hf_config['hidden_size']
    config['model']['attention']['n_in'] = hf_config['hidden_size']
    config['model']['attention']['n_heads'] = hf_config['num_attention_heads']
    config['model']['attention']['n_query_groups'] = hf_config['num_key_value_heads']
    config['model']['feedforward']['n_in'] = hf_config['hidden_size']
    config['model']['feedforward']['n_hidden'] = hf_config['intermediate_size']
    config['model']['head']['n_in'] = hf_config['hidden_size']
    config['model']['head']['n_out'] = hf_config['vocab_size']
    config['model']['norm']['n_in'] = hf_config['hidden_size']
    
    sd = {}
    wmap = WeightMap.get_llama2_weight_map(num_blocks=config['model']['n_blocks'])
    
    for file in files:
        weights = torch.load(file)
        for key in weights:
            if key not in wmap:
                continue
            sd[wmap[key]] = weights[key]
    
    fabric = Fabric(devices=1, accelerator='gpu', precision='16-true')
    fabric.print(config)
    with fabric.init_module(empty_init=True):
        model = registry.resolve(config)['model']
    model.load_state_dict(state_dict=sd, strict=True)
    
    tokenizer = Tokenizer(checkpoint_dir=checkpoint_dir)
    
    state = {'model': model, 'tokenizer': tokenizer, 'config': config}
    
    fabric.save(state=state, path=save_path)
    
if __name__ == "__main__":
    CLI(convert_hf_checkpoint)