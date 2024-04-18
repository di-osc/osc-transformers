from jsonargparse import CLI 
from osc_transformers.model_helpers.huggingface import get_supported_architectures
from osc_transformers.model_helpers.huggingface.base import HFModelHelper
from osc_transformers.config import registry
from pathlib import Path
from wasabi import msg
from typing import Literal
import json


def get_hf_model_helper(checkpoint_dir: str) -> HFModelHelper:
    config_path = Path(checkpoint_dir) / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    architecture = config["architectures"][0]
    allowed_architectures = get_supported_architectures()
    if architecture not in allowed_architectures:
        msg.fail(title="Architecture {architecture} is not supported.",
                 text=f"Supported architectures are: {allowed_architectures}", 
                 exits=1)
    model_helper: HFModelHelper = registry.model_helpers.get(architecture)(checkpoint_dir)
    return model_helper

def convert(checkpoint_dir: str, 
            config_name: str = 'config.cfg', 
            model_name: str = 'osc_model.pth'):
    """Convert a huggingface checkpoint to osc_transformers checkpoint.

    Args:
        checkpoint_dir: Path to the directory containing the checkpoint.
        config_name: Name of the config file. Default is 'config.cfg'.
        model_name: Name of the model file. Default is 'osc_model.pth'.
    """
    model_helper = get_hf_model_helper(checkpoint_dir)
    model_helper.convert_checkpoint(config_name=config_name, model_name=model_name)
    
def quantize_int8(checkpoint_dir: str, save_dir: str):
    """
    Quantize the model to int8.
    Args:
        checkpoint_dir: Path to the directory containing the checkpoint.
        save_dir: Path to the directory to save the quantized model.
    """
    model_helper = get_hf_model_helper(checkpoint_dir)
    model_helper.quantize_int8(save_dir=save_dir)
    
def quantize_int4(checkpoint_dir: str, 
                  save_dir: str, 
                  groupsize: Literal[32, 64, 128, 256] = 32, 
                  k: Literal[2, 4, 8] = 8, 
                  padding: bool = True, 
                  device: str = 'cuda:0'):
    """
    Quantize the model to int4.
    Args:
        checkpoint_dir: Path to the directory containing the checkpoint.
        save_dir: Path to the directory to save the quantized model.
        groupsize: The groupsize to use for the quantization.
        k: The k parameter to use for the quantization.
        padding: Whether to pad the model.
        device: The device to use for the quantization.
    """
    model_helper = get_hf_model_helper(checkpoint_dir)
    model_helper.quantize_int4(save_dir=save_dir, groupsize=groupsize, k=k, padding=padding, device=device)
    
    
commands= {
    "convert": convert,
    "quantize": {
        "int8": quantize_int8,
        "int4": quantize_int4
    }
}

def main():
    CLI(components=commands)