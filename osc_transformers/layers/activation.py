from ..config import registry
import torch.nn as nn 



@registry.layers.register("SiLU")
def SiLU(inplace: bool = False) -> nn.Module:
    return nn.SiLU(inplace=inplace)


@registry.layers.register("GeLU")
def GELU() -> nn.Module:
    return nn.GELU()


@registry.layers.register("ReLU")
def ReLU(inplace: bool = False) -> nn.Module:
    return nn.ReLU(inplace=inplace)