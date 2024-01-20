from .layers import *
from .architectures import *
from .models import *
from .config import registry, Config
from .tokenizer import Tokenizer




def build_model(name: str):
    """Build a model from a name."""
    config = registry.configs.get(name)()
    model = registry.resolve(config)["model"]
    return model