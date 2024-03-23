from pathlib import Path
from typing import Dict, Union
from ..config import Config, registry



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