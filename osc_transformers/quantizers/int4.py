from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn

from ..registry import Registry
from .base import Quantizer


@Registry.quantizers.register("int4")
class WeightOnlyInt4Quantizer(Quantizer):
    def __init__(
        self,
        groupsize: Literal[32, 64, 128, 256] = 32,
        inner_k_tiles: Literal[2, 4, 8] = 8,
        padding_allowed: bool = True,
    ):
        self.groupsize = groupsize
        self.inner_k_tiles = inner_k_tiles
        self.padding_allowed = padding_allowed
        assert groupsize in [32, 64, 128, 256]
        assert inner_k_tiles in [2, 4, 8]

    def quantize(self, model: nn.Module) -> nn.Module:
        for name, children in model.named_children():
            if isinstance(children, torch.nn.Linear):
                out_features = children.out_features
                in_features = children.in_features
                assert not children.bias
                assert out_features % 8 == 0, "require out_features % 8 == 0"
                weight = children.weight.data
                if not _check_linear_int4_k(in_features, self.groupsize, self.inner_k_tiles):
                    if self.padding_allowed:
                        import torch.nn.functional as F

                        print(f"warning: {name} is padded to satisfy in_features % 1024 == 0")
                        padded_in_features = find_multiple(in_features, 1024)
                        weight = F.pad(weight, pad=(0, padded_in_features - in_features))
                        in_features = padded_in_features
                    else:
                        print(
                            f"warning: {name} is skipped, int4 requires that in_features is 32, 64, or is divisible by 1024, "
                            + "and that groupsize and inner_k_tiles*16 evenly divide into it"
                        )
                        continue
                weight_int4pack, scales_and_zeros = prepare_int4_weight_and_scales_and_zeros(
                    weight.to(torch.bfloat16), self.groupsize, self.inner_k_tiles
                )
                int4_linear = WeightOnlyInt4Linear(
                    in_features=in_features,
                    out_features=out_features,
                    groupsize=self.groupsize,
                    inner_k_tiles=self.inner_k_tiles,
                )
                int4_linear.weight = weight_int4pack
                int4_linear.scales_and_zeros = scales_and_zeros
                setattr(model, name, int4_linear)
            else:
                self.quantize(children)
        return model

    @torch.no_grad()
    def save_quantized_state_dict(self, model: nn.Module, save_path: str | Path):
        cur_state_dict = model.state_dict()
        for fqn, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear):
                assert not mod.bias
                out_features = mod.out_features
                in_features = mod.in_features
                assert out_features % 8 == 0, "require out_features % 8 == 0"
                weight = mod.weight.data
                if not _check_linear_int4_k(in_features, self.groupsize, self.inner_k_tiles):
                    if self.padding_allowed:
                        import torch.nn.functional as F

                        print(f"warning: {fqn} is padded to satisfy in_features % 1024 == 0")
                        padded_in_features = find_multiple(in_features, 1024)
                        weight = F.pad(weight, pad=(0, padded_in_features - in_features))
                    else:
                        print(
                            f"warning: {fqn} is skipped, int4 requires that in_features is 32, 64, or is divisible by 1024, "
                            + "and that groupsize and inner_k_tiles*16 evenly divide into it"
                        )
                        continue
                weight_int4pack, scales_and_zeros = prepare_int4_weight_and_scales_and_zeros(
                    weight.to(torch.bfloat16), self.groupsize, self.inner_k_tiles
                )
                cur_state_dict[f"{fqn}.weight"] = weight_int4pack.to("cpu")
                cur_state_dict[f"{fqn}.scales_and_zeros"] = scales_and_zeros.to("cpu")
                if cur_state_dict.get(f"{fqn}.bias") is not None:
                    # remove bias
                    del cur_state_dict[f"{fqn}.bias"]

        save_path = Path(save_path)
        if save_path.exists():
            save_path.unlink()
        torch.save(cur_state_dict, save_path)

    def convert_for_runtime(self, model: nn.Module, use_cuda: bool = True):
        replace_linear_int4(model, self.groupsize, self.inner_k_tiles, self.padding_allowed, use_cuda)
        return model


def _check_linear_int4_k(k, groupsize=1, inner_k_tiles=1):
    return k % groupsize == 0 and k % (inner_k_tiles * 16) == 0


def replace_linear_int4(module, groupsize, inner_k_tiles, padding_allowed, use_cuda):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            if _check_linear_int4_k(child.in_features, groupsize, inner_k_tiles) or padding_allowed:
                new_module = WeightOnlyInt4Linear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=False,
                    groupsize=groupsize,
                    inner_k_tiles=inner_k_tiles,
                )
                setattr(module, name, new_module)
        else:
            replace_linear_int4(child, groupsize, inner_k_tiles, padding_allowed, use_cuda)


def prepare_int4_weight_and_scales_and_zeros(weight_bf16, groupsize, inner_k_tiles):
    weight_int32, scales_and_zeros = group_quantize_tensor(weight_bf16, n_bit=4, groupsize=groupsize)
    weight_int4pack = torch.ops.aten._convert_weight_to_int4pack(weight_int32, inner_k_tiles)
    return weight_int4pack, scales_and_zeros


def group_quantize_tensor(w: nn.Module, n_bit: int = 4, groupsize: int = 128):
    scales, zeros = get_group_qparams(w, n_bit, groupsize)
    w_int32 = group_quantize_tensor_from_qparams(w, scales, zeros, n_bit, groupsize)
    scales_and_zeros = pack_scales_and_zeros(scales, zeros)
    return w_int32, scales_and_zeros


def get_group_qparams(w, n_bit=4, groupsize=128):
    # needed for GPTQ with padding
    if groupsize > w.shape[-1]:
        groupsize = w.shape[-1]
    assert groupsize > 1
    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2

    to_quant = w.reshape(-1, groupsize)
    assert torch.isnan(to_quant).sum() == 0

    max_val = to_quant.amax(dim=1, keepdim=True)
    min_val = to_quant.amin(dim=1, keepdim=True)
    max_int = 2**n_bit - 1
    scales = (max_val - min_val).clamp(min=1e-6) / max_int
    zeros = min_val + scales * (2 ** (n_bit - 1))
    return scales.to(torch.bfloat16).reshape(w.shape[0], -1), zeros.to(torch.bfloat16).reshape(w.shape[0], -1)


def pack_scales_and_zeros(scales, zeros):
    assert scales.shape == zeros.shape
    assert scales.dtype == torch.bfloat16
    assert zeros.dtype == torch.bfloat16
    return (
        torch.cat(
            [
                scales.reshape(scales.size(0), scales.size(1), 1),
                zeros.reshape(zeros.size(0), zeros.size(1), 1),
            ],
            2,
        )
        .transpose(0, 1)
        .contiguous()
    )


def unpack_scales_and_zeros(scales_and_zeros):
    assert len(scales_and_zeros.shape) == 3 and scales_and_zeros.shape[2] == 2
    assert scales_and_zeros.dtype == torch.float
    return torch.split(scales_and_zeros.transpose(0, 1), 1, 2)


def group_quantize_tensor_from_qparams(w, scales, zeros, n_bit=4, groupsize=128):
    assert groupsize > 1
    # needed for GPTQ single column quantize
    if groupsize > w.shape[-1] and scales.shape[-1] == 1:
        groupsize = w.shape[-1]

    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2

    to_quant = w.reshape(-1, groupsize)
    assert torch.isnan(to_quant).sum() == 0

    scales = scales.reshape(-1, 1)
    zeros = zeros.reshape(-1, 1)
    min_val = zeros - scales * (2 ** (n_bit - 1))
    max_int = 2**n_bit - 1
    min_int = 0
    w_int32 = to_quant.sub(min_val).div(scales).round().clamp_(min_int, max_int).to(torch.int32).reshape_as(w)
    return w_int32


class WeightOnlyInt4Linear(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias=False,
        groupsize: int = 128,
        inner_k_tiles: int = 8,
    ) -> None:
        super().__init__()
        self.padding = not self._check_linear_int4_k(in_features, groupsize, inner_k_tiles)
        if self.padding:
            self.origin_in_features = in_features
            in_features = find_multiple(in_features, 1024)

        self.in_features = in_features
        self.out_features = out_features
        assert not bias, "require bias=False"
        self.groupsize = groupsize
        self.inner_k_tiles = inner_k_tiles

        assert out_features % 8 == 0, "require out_features % 8 == 0"
        assert in_features % (inner_k_tiles * 16) == 0, "require in_features % (innerKTiles * 16) == 0"
        self.register_buffer(
            "weight",
            torch.empty(
                (
                    out_features // 8,
                    in_features // (inner_k_tiles * 16),
                    32,
                    inner_k_tiles // 2,
                ),
                dtype=torch.int32,
            ),
        )
        self.register_buffer(
            "scales_and_zeros",
            torch.empty((in_features // groupsize, out_features, 2), dtype=torch.bfloat16),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.to(torch.bfloat16)
        if self.padding:
            import torch.nn.functional as F

            input = F.pad(input, pad=(0, self.in_features - self.origin_in_features))
        return self._linear_forward_int4(input, self.weight, self.scales_and_zeros, self.out_features, self.groupsize)

    def _check_linear_int4_k(self, in_features: int, groupsize: int, inner_k_tiles: int) -> bool:
        """check if the input features are compatible with the linear int4 kernel

        Args:
            in_features (int): the number of input features
            groupsize (int): the group size
            inner_k_tiles (int): the number of inner k tiles
        """
        return in_features % (inner_k_tiles * 16) == 0 and in_features % groupsize == 0

    def _linear_forward_int4(self, x, weight_int4pack, scales_and_zeros, out_features, groupsize):
        origin_x_size = x.size()
        x = x.reshape(-1, origin_x_size[-1])
        c = torch.ops.aten._weight_int4pack_mm(x, weight_int4pack, groupsize, scales_and_zeros)
        new_shape = origin_x_size[:-1] + (out_features,)
        c = c.reshape(new_shape)
        return c


def find_multiple(n: int, k: int) -> int:
    assert k > 0
    if n % k == 0:
        return n
    return n + k - (n % k)
