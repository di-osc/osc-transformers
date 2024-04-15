from abc import ABC, abstractmethod
import torch.nn as nn
from confection import Config



class Quantizer(ABC):
    
    @abstractmethod
    def convert_for_runtime(self, model: nn.Module) -> nn.Module:
        """Converts the original model to a quantized model for runtime.

        Args:
            model (nn.Module): The original model.
        """
        raise NotImplementedError
    
    @abstractmethod
    def save_quantized_state_dict(self, model: nn.Module, save_path: str) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def get_quantizer_config(self) -> Config:
        raise NotImplementedError