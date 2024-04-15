import json 
from pathlib import Path
from typing import Dict
import torch
from typing import Literal
from osc_transformers.config import Config
from osc_transformers.model_helpers.build import build_from_config
from osc_transformers.quantizers import WeightOnlyInt4Quantizer, WeightOnlyInt8Quantizer



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
        helper = WeightOnlyInt8Quantizer()
        helper.save_quantized_state_dict(model=model, save_path=self.checkpoint_dir / save_name)
        
    def quantize_int4(self, 
                      groupsize: Literal[32, 64, 128, 256] = 32, 
                      k: Literal[2, 4, 8] = 8, 
                      padding: bool = True,
                      save_name: str = 'osc_model_int4-G{groupsize}-K{k}.pth', 
                      device: str = 'cuda'):
        assert torch.cuda.is_available(), 'Only support cuda device for int4 quantization'
        assert 'cuda' in device, 'Only support cuda device for int4 quantization'
        save_name = save_name.format(groupsize=groupsize, k=k)
        model = self.load_checkpoint(device=device)
        helper = WeightOnlyInt4Quantizer(groupsize=groupsize, inner_k_tiles=k, padding_allowed=padding)
        helper.save_quantized_state_dict(model=model, save_path=self.checkpoint_dir / save_name)