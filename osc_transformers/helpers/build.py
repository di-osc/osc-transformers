from pathlib import Path
from typing import Dict, Union
from ..config import Config, registry
import torch



def build_from_config(config: Union[Dict, str, Path, Config], model_section: str = 'model'):
    """Build a model from a configuration.

    Args:
        config (Union[Dict, str, Path, Config]): the configuration to build the model from, can be a dictionary, a path to a file or a Config object.
        model_section (str, optional): the section to look for the model in the configuration. Defaults to 'model'.

    Returns:
        torch.nn.Module: the model built from the configuration.
    """
    if isinstance(config, (str, Path)):
        config = Config().from_disk(config)
    if isinstance(config, dict):
        config = Config(data=config)
    return registry.resolve(config=config)[model_section]


def load_from_checkpoint(checkpoint_dir: Union[str, Path], model_section: str = 'model', config_name: str = 'config.cfg', model_name: str = 'osc_model.pth'):
    """Load a model from a checkpoint directory.

    Args:
        checkpoint_dir (Union[str, Path]): the directory containing the model checkpoint.
        model_section (str, optional): the section to look for the model in the configuration. Defaults to 'model'.
        config_name (str, optional): the name of the configuration file. Defaults to 'config.cfg'.
        model_name (str, optional): the name of the model file. Defaults to 'osc_model.pth'.

    Returns:
        torch.nn.Module: the model loaded from the checkpoint.
    """
    checkpoint_dir = Path(checkpoint_dir)
    config_path = Path(checkpoint_dir) / config_name
    model = build_from_config(config_path, model_section=model_section)
    states = torch.load(str(checkpoint_dir / model_name), map_location='cpu', mmap=True, weights_only=True)
    model.load_state_dict(states)
    return model