from jsonargparse import CLI 
from osc_transformers.model_helpers.huggingface import get_supported_architectures
from osc_transformers.model_helpers.huggingface.base import HFModelHelper
from osc_transformers.config import registry
from pathlib import Path
from wasabi import msg
import json

def convert(checkpoint_dir: str, 
            config_name: str = 'config.cfg', 
            model_name: str = 'osc_model.pth'):
    """Convert a huggingface checkpoint to osc_transformers checkpoint.

    Args:
        checkpoint_dir: Path to the directory containing the checkpoint.
        config_name: Name of the config file. Default is 'config.cfg'.
        model_name: Name of the model file. Default is 'osc_model.pth'.
    """
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
    model_helper.convert_checkpoint(config_name=config_name, model_name=model_name)
    
    
commands= {
    "convert": convert,
}

def main():
    CLI(components=commands)