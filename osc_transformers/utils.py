from pathlib import Path
from typing import Dict, Union
from osc_transformers.config import Config, registry
from osc_transformers.quantizers.base import Quantizer
import torch
import statistics
from wasabi import msg



def build_model(config: Union[Dict, str, Path, Config], 
                model_section: str = 'model',
                quantizer_section: str = 'quantizer',
                empty_init: bool = True,
                quantize: bool = True):
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
    if empty_init:
        with torch.device('meta'):
            resolved = registry.resolve(config=config)
    else:
        resolved = registry.resolve(config=config)[model_section]
    if model_section not in resolved:
        msg.fail(f"cannot find model section {model_section}")
    else:
        model = resolved[model_section]
    if quantizer_section in resolved and quantize:
        quantizer: Quantizer = resolved[quantizer_section]
        model = quantizer.convert_for_runtime(model=model)
    return model


def buil_from_checkpoint(checkpoint_dir: Union[str, Path], 
                         model_section: str = 'model', 
                         config_name: str = 'config.cfg', 
                         model_name: str = 'osc_model.pth',
                         empty_init: bool = True,
                         quantize: bool = True,
                         load_weights_only: bool = True,
                         load_weights: bool = True):
    """build a model from a checkpoint directory.

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
    with torch.device('meta'):
        model = build_model(config_path, 
                            model_section=model_section,
                            quantize=quantize,
                            empty_init=empty_init)
    if load_weights:
        states = torch.load(str(checkpoint_dir / model_name), map_location='cpu', mmap=True, weights_only=load_weights_only)
        model.load_state_dict(states)
    return model


@torch.no_grad()
def benchmark(model, input, num_iters=10):
    """Runs the model on the input several times and returns the median execution time."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(num_iters):
        start.record()
        model(input)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1000)
    return statistics.median(times)


def find_multiple(n: int, k: int) -> int:
    """Find the smallest multiple of k that is greater than or equal to n.

    Args:
        n (int): the number to find the multiple of k for.
        k (int): the number to find the multiple of.

    Returns:
        int: the smallest multiple of k that is greater than or equal to n.
    """
    if n % k == 0:
        return n
    return n + k - (n % k)