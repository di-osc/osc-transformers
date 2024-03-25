import json 
from pathlib import Path
from typing import Dict
from ..config import Config
import torch



class HFModelHelper:
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        with open(self.checkpoint_dir / "config.json", "r") as f:
            self.hf_config = json.load(f)
        assert self.hf_architecture in self.hf_config['architectures'], f'Only support {self.hf_architecture} model, current model is {self.hf_config["architectures"]}'
    
    @property
    def weight_map(self) -> Dict:
        raise NotImplementedError("Method not implemented")
    
    @property
    def hf_architecture(self) -> str:
        raise NotImplementedError("Method not implemented")
    
    @property
    def osc_config(self) -> Config:
        raise NotImplementedError("Method not implemented")
    
    def convert_pytorch_format(self, config_name: str = 'config.cfg', model_name: str = 'osc_model.pth'):
        sd = {}
        wmap = self.weight_map
        index_file = self.checkpoint_dir / 'pytorch_model.bin.index.json'
        if index_file.exists():
            with open(index_file, 'r') as f:
                index = json.load(f)
            files = [self.checkpoint_dir / file  for file in set(index['weight_map'].values())]
        else:
            files = [self.checkpoint_dir / 'pytorch_model.bin']
        assert len(files) > 0, 'No pytorch model file found'
        for file in files:
            weights = torch.load(str(file), map_location='cpu', weights_only=True, mmap=True)
            for key in weights:
                if key not in wmap:
                    continue
                sd[wmap[key]] = weights[key]
            
        self.osc_config.to_disk(self.checkpoint_dir / config_name)
        torch.save(sd, self.checkpoint_dir / model_name)
        
    def convert_safetensor_format(self, config_name: str = 'config.cfg', model_name: str = 'osc_model.pth'):
        sd = {}
        wmap = self.weight_map
        index_file = self.checkpoint_dir / 'model.safetensors.index.json'
        if index_file.exists():
            with open(index_file, 'r') as f:
                index = json.load(f)
            files = [self.checkpoint_dir / file  for file in set(index['weight_map'].values())]
        else:
            files = [self.checkpoint_dir / 'model.safetensors']
        assert len(files) > 0, 'No pytorch model file found'
        try:
            from safetensors import safe_open
        except:
            raise ImportError("Please install safetensors first, run `pip install safetensors`")
        for file in files:
            with safe_open(file, framework='pt') as f:
                for key in f.keys():
                    if key not in wmap:
                        continue
                    sd[wmap[key]] = f.get_tensor(key)
            
        self.osc_config.to_disk(self.checkpoint_dir / config_name)
        torch.save(sd, self.checkpoint_dir / model_name)
    
    
class Llama2Helper(HFModelHelper):
        
    @property
    def weight_map(self) -> Dict:
        """获取llama2权重映射表
        """
        weight_map = {
        "model.embed_tokens.weight": "embedding.embed.weight",
        "model.norm.weight": "head_norm.weight",
        "lm_head.weight": "head.weight",
        }
        
        for i in range(self.hf_config['num_hidden_layers']):
            weight_map[f"model.layers.{i}.input_layernorm.weight"] = f"blocks.{i}.attention_norm.weight"
            weight_map[f"model.layers.{i}.post_attention_layernorm.weight"] = f"blocks.{i}.feedforward_norm.weight"
            weight_map[f"model.layers.{i}.self_attn.q_proj.weight"] = f"blocks.{i}.attention.q_proj.weight"
            weight_map[f"model.layers.{i}.self_attn.k_proj.weight"] = f"blocks.{i}.attention.k_proj.weight"
            weight_map[f"model.layers.{i}.self_attn.v_proj.weight"] = f"blocks.{i}.attention.v_proj.weight"
            weight_map[f"model.layers.{i}.self_attn.o_proj.weight"] = f"blocks.{i}.attention.o_proj.weight"
            weight_map[f"model.layers.{i}.mlp.gate_proj.weight"] = f"blocks.{i}.feedforward.gate_proj.weight"
            weight_map[f"model.layers.{i}.mlp.up_proj.weight"] = f"blocks.{i}.feedforward.up_proj.weight"
            weight_map[f"model.layers.{i}.mlp.down_proj.weight"] = f"blocks.{i}.feedforward.down_proj.weight"
            
        return weight_map
    
    @property
    def hf_architecture(self) -> str:
        return "LlamaForCausalLM"
    
    @property
    def osc_config(self) -> Config:
        tempelate = """
        [model]
        @architectures = "TransformerDecoder"
        n_blocks = {num_hidden_layers}
        block_size = {max_length}
        prenorm = "True"

        [model.attention]
        @layers = "SelfAttention"
        n_in = {hidden_size}
        n_heads = {num_attention_heads}
        n_query_groups = {num_key_value_heads}
        q_bias = "False"
        k_bias = "False"
        v_bias = "False"
        o_bias = "False"

        [model.embedding]
        @layers = "TokenEmbedding"
        n_embeddings = {vocab_size}
        embedding_size = {hidden_size}

        [model.feedforward]
        @layers = "SwiGLU"
        n_in = {hidden_size}
        n_hidden = {intermediate_size}

        [model.head]
        @layers = "Linear"
        n_in = {hidden_size}
        n_out = {vocab_size}
        bias = "False"

        [model.norm]
        @layers = "RMSNorm"
        n_in = {hidden_size}
        eps = 0.00001
        """
        self.hf_config['max_length'] = self.hf_config.get('max_length', self.hf_config['max_position_embeddings'])
        config_str = tempelate.format(**self.hf_config)
        return Config().from_str(config_str)
    
    
class Qwen2Helper(HFModelHelper):
    
    @property
    def weight_map(self) -> Dict:
        """获取qwen2权重映射表
        """
        weight_map = {
            "model.embed_tokens.weight": "embedding.embed.weight",
            "model.norm.weight": "head_norm.weight",
            "lm_head.weight": "head.weight",
        }
        
        for i in range(self.hf_config['num_hidden_layers']):
            weight_map[f"model.layers.{i}.input_layernorm.weight"] = f"blocks.{i}.attention_norm.weight"
            weight_map[f"model.layers.{i}.post_attention_layernorm.weight"] = f"blocks.{i}.feedforward_norm.weight"
            weight_map[f"model.layers.{i}.self_attn.q_proj.weight"] = f"blocks.{i}.attention.q_proj.weight"
            weight_map[f"model.layers.{i}.self_attn.q_proj.bias"] = f"blocks.{i}.attention.q_proj.bias"
            weight_map[f"model.layers.{i}.self_attn.k_proj.weight"] = f"blocks.{i}.attention.k_proj.weight"
            weight_map[f"model.layers.{i}.self_attn.k_proj.bias"] = f"blocks.{i}.attention.k_proj.bias"
            weight_map[f"model.layers.{i}.self_attn.v_proj.weight"] = f"blocks.{i}.attention.v_proj.weight"
            weight_map[f"model.layers.{i}.self_attn.v_proj.bias"] = f"blocks.{i}.attention.v_proj.bias"
            weight_map[f"model.layers.{i}.self_attn.o_proj.weight"] = f"blocks.{i}.attention.o_proj.weight"
            weight_map[f"model.layers.{i}.mlp.gate_proj.weight"] = f"blocks.{i}.feedforward.gate_proj.weight"
            weight_map[f"model.layers.{i}.mlp.up_proj.weight"] = f"blocks.{i}.feedforward.up_proj.weight"
            weight_map[f"model.layers.{i}.mlp.down_proj.weight"] = f"blocks.{i}.feedforward.down_proj.weight"
            
        return weight_map
    
    @property
    def hf_architecture(self) -> str:
        return "Qwen2ForCausalLM"
    
    @property
    def osc_config(self) -> Config:
        tempelate = """
        [model]
        @architectures = "TransformerDecoder"
        n_blocks = {num_hidden_layers}
        block_size = {max_length}
        prenorm = "True"

        [model.attention]
        @layers = "SelfAttention"
        n_in = {hidden_size}
        n_heads = {num_attention_heads}
        n_query_groups = {num_key_value_heads}
        q_bias = "True"
        k_bias = "True"
        v_bias = "True"
        o_bias = "False"

        [model.embedding]
        @layers = "TokenEmbedding"
        n_embeddings = {vocab_size}
        embedding_size = {hidden_size}

        [model.feedforward]
        @layers = "SwiGLU"
        n_in = {hidden_size}
        n_hidden = {intermediate_size}

        [model.head]
        @layers = "Linear"
        n_in = {hidden_size}
        n_out = {vocab_size}
        bias = "False"

        [model.norm]
        @layers = "RMSNorm"
        n_in = {hidden_size}
        eps = 0.000001
        """
        self.hf_config['max_length'] = self.hf_config.get('max_length', self.hf_config['max_position_embeddings'])
        config_str = tempelate.format(**self.hf_config)
        return Config().from_str(config_str)