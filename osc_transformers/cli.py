from jsonargparse import CLI 
from osc_transformers.model_helpers.huggingface import get_supported_architectures
from osc_transformers.model_helpers.huggingface.base import HFModelHelper
from osc_transformers.quantizers import WeightOnlyInt8Quantizer, WeightOnlyInt4Quantizer
from osc_transformers.tokenizer import Tokenizer
from osc_transformers.utils import buil_from_checkpoint
from osc_transformers.config import registry
from pathlib import Path
from wasabi import msg
from typing import Literal, Optional
import json
import torch


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


def convert(checkpoint_dir: str, save_dir: Optional[str] = None):
    """Convert a huggingface checkpoint to osc_transformers checkpoint.

    Args:
        checkpoint_dir: Path to the directory containing the checkpoint.
        save_dir: Path to the directory to save the converted checkpoint. if None, the converted checkpoint will be saved in the same directory as the original checkpoint.
    """
    model_helper = get_hf_model_helper(checkpoint_dir)
    if not save_dir:
        save_dir = checkpoint_dir
    model_helper.convert_checkpoint(save_dir=save_dir)
    
    
def quantize_int8(checkpoint_dir: str, save_dir: str):
    """
    Quantize the osc model to int8.
    
    Args:
        checkpoint_dir: Path to the osc model directory containing the checkpoint.
        save_dir: Path to the directory to save the quantized model.
    """
    save_dir= Path(save_dir)
    if save_dir == checkpoint_dir:
        msg.warn("The quantized model will replace the original model.")
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    tokenizer = Tokenizer(checkpoint_dir=checkpoint_dir)
    model, config = buil_from_checkpoint(checkpoint_dir=checkpoint_dir, return_config=True)
    quantizer = WeightOnlyInt8Quantizer()
    model = quantizer.quantize(model)
    config = config.merge(quantizer.quantizer_config)
    torch.save(model.state_dict(), Path(save_dir) / "osc_model.pth")
    config.to_disk(Path(save_dir) / "config.cfg")
    tokenizer.save(save_dir)
    
    
def quantize_int4(checkpoint_dir: str, 
                  save_dir: str, 
                  groupsize: Literal[32, 64, 128, 256] = 32, 
                  k: Literal[2, 4, 8] = 8, 
                  padding: bool = True, 
                  device: str = 'cuda:0'):
    """
    Quantize the osc model to int4.
    
    Args:
        checkpoint_dir: Path to the osc model directory containing the checkpoint.
        save_dir: Path to the directory to save the quantized model.
        groupsize: The groupsize to use for the quantization.
        k: The k parameter to use for the quantization.
        padding: Whether to pad the model.
        device: The device to use for the quantization.
    """
    save_dir = Path(save_dir)
    if save_dir == checkpoint_dir:
        msg.warn("The quantized model will replace the original model.")
    if not Path(save_dir).exists():
        Path(save_dir).mkdir(parents=True)
    tokenizer = Tokenizer(checkpoint_dir=checkpoint_dir)
    model, config = buil_from_checkpoint(checkpoint_dir=checkpoint_dir, return_config=True)
    model.to(device)
    quantizer = WeightOnlyInt4Quantizer(groupsize=groupsize, inner_k_tiles=k, padding_allowed=padding)
    model = quantizer.quantize(model)
    config = config.merge(quantizer.quantizer_config)
    torch.save(model.state_dict(), Path(save_dir) / "osc_model.pth")
    config.to_disk(Path(save_dir) / "config.cfg")
    tokenizer.save(save_dir)
    
    
commands= {
    "convert": convert,
    "quantize": {
        "int8": quantize_int8,
        "int4": quantize_int4
    }
}

def main():
    CLI(components=commands)