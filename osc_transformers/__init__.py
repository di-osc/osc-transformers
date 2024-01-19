from .layers import *
from .architectures import TransformerDecoder
from .models import *
from .config import registry, Config
from .utils import WeightMap
from .tokenizer import Tokenizer




def build_model(name: str):
    """Build a model from a name."""
    config = registry.configs.get(name)()
    model = registry.resolve(config)["model"]
    return model