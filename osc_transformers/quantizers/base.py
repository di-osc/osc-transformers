from abc import ABC, abstractmethod
import torch.nn as nn



class Quantizer(ABC):
    
    
    @abstractmethod
    def convert_for_runtime(self, model: nn.Module) -> nn.Module:
        raise NotImplementedError
    
    @abstractmethod
    def save_quantized_state_dict(self, model: nn.Module, save_path: str):
        raise NotImplementedError