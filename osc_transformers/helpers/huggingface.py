import json 
from pathlib import Path
from typing import Dict
from ..config import Config, registry
from .build import build_from_config
from .quantize import WeightOnlyInt4QuantHelper, WeightOnlyInt8QuantHelper
import torch
from typing import Literal



class HFModelHelper:
    
    hf_architecture: str
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        with open(self.checkpoint_dir / "config.json", "r") as f:
            self.hf_config = json.load(f)
        assert self.hf_architecture in self.hf_config['architectures'], f'Only support {self.hf_architecture} model, current model is {self.hf_config["architectures"]}'
    
    @property
    def weight_map(self) -> Dict:
        raise NotImplementedError("Method not implemented")
    
    @property
    def osc_config(self) -> Config:
        raise NotImplementedError("Method not implemented")
    
    def convert_checkpoint(self, config_name: str = 'config.cfg', model_name: str = 'osc_model.pth'):
        """将huggingface模型转换为osc格式模型

        Args:
            config_name (str, optional): 配置文件保存名称. Defaults to 'config.cfg'.
            model_name (str, optional): 模型文件名称. Defaults to 'osc_model.pth'.
        """
        pytorch_model = Path(self.checkpoint_dir) / 'pytorch_model.bin'
        pytorch_idx_file = Path(self.checkpoint_dir) / 'pytorch_model.bin.index.json'
        if pytorch_model.exists() or pytorch_idx_file.exists():
            self.convert_pytorch_format(config_name, model_name)
        safetensors_model = Path(self.checkpoint_dir) / 'model.safetensors'
        safetensors_idx_file = Path(self.checkpoint_dir) / 'model.safetensors.index.json'
        if safetensors_model.exists() or safetensors_idx_file.exists():
            self.convert_safetensor_format(config_name, model_name)
        if not pytorch_model.exists() and not safetensors_model.exists() and not pytorch_idx_file.exists() and not safetensors_idx_file.exists():
            raise FileNotFoundError("No pytorch model file found")
    
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
        except Exception:
            raise ImportError("Please install safetensors first, run `pip install safetensors`")
        for file in files:
            with safe_open(file, framework='pt') as f:
                for key in f.keys():
                    if key not in wmap:
                        continue
                    sd[wmap[key]] = f.get_tensor(key)
            
        self.osc_config.to_disk(self.checkpoint_dir / config_name)
        torch.save(sd, self.checkpoint_dir / model_name)
        
    def load_checkpoint(self, checkpoint_name: str = 'osc_model.pth', device: str = 'cuda', dtype: torch.dtype = torch.bfloat16):
        model = build_from_config(self.osc_config)
        model.load_state_dict(torch.load(str(self.checkpoint_dir / checkpoint_name), mmap=True, weights_only=True), assign=True)
        model.to(device, dtype=dtype)
        return model.eval()
        
    def quantize_int8(self, save_name: str = 'osc_model_int8.pth', device: str = 'cuda'):
        model = self.load_checkpoint(device=device)
        helper = WeightOnlyInt8QuantHelper(model=model)
        helper.save_quantized_state_dict(self.checkpoint_dir / save_name)
        
    def quantize_int4(self, 
                      groupsize: Literal[32, 64, 128, 256] = 32, 
                      k: Literal[2, 4, 8] = 8, 
                      padding: bool = True,
                      save_name: str = 'osc_model_int4.pth', 
                      device: str = 'cuda'):
        assert 'cuda' in device, 'Only support cuda device for int4 quantization'
        model = self.load_checkpoint(device=device)
        helper = WeightOnlyInt4QuantHelper(model=model, groupsize=groupsize, inner_k_tiles=k, padding_allowed=padding)
        helper.save_quantized_state_dict(self.checkpoint_dir / save_name)
    

@registry.model_helpers.register("LlamaForCausalLM")
class LlamaHelper(HFModelHelper):
        
    hf_architecture = "LlamaForCausalLM"
    
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
    def osc_config(self) -> Config:
        tempelate = """
        [model]
        @architectures = "TransformerDecoder"
        n_blocks = {num_hidden_layers}
        block_size = {max_length}
        prenorm = "True"
        rope_base = {rope_theta}

        [model.attention]
        @layers = "CausalSelfAttention"
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
        eps = {rms_norm_eps}
        """
        self.hf_config['max_length'] = self.hf_config.get('max_length', self.hf_config['max_position_embeddings'])
        self.hf_config['rope_theta'] = self.hf_config.get('rope_theta', 10000)
        self.hf_config['rms_norm_eps'] = self.hf_config.get('rms_norm_eps', 1e-5)
        config_str = tempelate.format(**self.hf_config)
        return Config().from_str(config_str)
    

@registry.model_helpers.register("Qwen2ForCausalLM")
class Qwen2Helper(HFModelHelper):
    
    hf_architecture = "Qwen2ForCausalLM"
    
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
    def osc_config(self) -> Config:
        tempelate = """
        [model]
        @architectures = "TransformerDecoder"
        n_blocks = {num_hidden_layers}
        block_size = {max_length}
        prenorm = "True"
        rope_base = {rope_theta}

        [model.attention]
        @layers = "CausalSelfAttention"
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