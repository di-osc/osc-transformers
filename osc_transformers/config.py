import catalogue
import confection
from confection import Config


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
    
    model_helpers = catalogue.create(
        "osc", 
        "model_helpers", 
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


__all__ = ["Config", "registry"]