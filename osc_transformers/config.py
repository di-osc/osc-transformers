import catalogue
import confection
from confection import Config
from typing import Union, Dict
from pathlib import Path


class registry(confection.registry):
    
    layers = catalogue.create(
        "osc", 
        "layers", 
        entry_points=True
    )
    
    architectures = catalogue.create(
        "osc", 
        "architectures", 
        entry_points=True
    )

    @classmethod
    def create(cls, registry_name: str, entry_points: bool = False) -> None:
        """Create a new custom registry."""
        if hasattr(cls, registry_name):
            raise ValueError(f"Registry '{registry_name}' already exists")
        reg = catalogue.create(
            "osc", registry_name, entry_points=entry_points
        )
        setattr(cls, registry_name, reg)


def build_model(config: Union[Dict, str, Path, Config], model_section: str = 'model'):
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


__all__ = ["Config", "registry", "build_model"]