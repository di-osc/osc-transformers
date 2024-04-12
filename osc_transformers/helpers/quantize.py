import torch 
import torch.nn as nn
from typing import Union
from pathlib import Path
from osc_transformers.layers.linear import WeightOnlyInt8Linear, WeightOnlyInt4Linear, find_multiple



def dynamically_quantize_per_channel(x, quant_min, quant_max, target_dtype):
    # assumes symmetric quantization
    # assumes axis == 0
    # assumes dense memory format
    # TODO(future): relax ^ as needed

    # default setup for affine quantization of activations
    eps = torch.finfo(torch.float32).eps

    # get min and max
    min_val, max_val = torch.aminmax(x, dim=1)

    # calculate scales and zero_points based on min and max
    # reference: https://fburl.com/code/srbiybme
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    device = min_val_neg.device

    # reference: https://fburl.com/code/4wll53rk
    max_val_pos = torch.max(-min_val_neg, max_val_pos)
    scales = max_val_pos / (float(quant_max - quant_min) / 2)
    # ensure scales is the same dtype as the original tensor
    scales = torch.clamp(scales, min=eps).to(x.dtype)
    zero_points = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)

    # quantize based on qmin/qmax/scales/zp
    # reference: https://www.internalfb.com/code/fbsource/[8edc275012b1]/fbcode/caffe2/torch/ao/quantization/fx/_decomposed.py?lines=63
    x_div = x / scales.unsqueeze(-1)
    x_round = torch.round(x_div)
    x_zp = x_round + zero_points.unsqueeze(-1)
    quant = torch.clamp(x_zp, quant_min, quant_max).to(target_dtype)

    return quant, scales, zero_points


def replace_linear_weight_only_int8_per_channel(module: nn.Module):
    """递归替换module中的所有nn.Linear为WeightOnlyInt8Linear"""
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, WeightOnlyInt8Linear(child.in_features, child.out_features))
        else:
            replace_linear_weight_only_int8_per_channel(child)


class WeightOnlyInt8QuantHelper:
    def __init__(self, model: nn.Module):
        self.model = model

    @torch.no_grad()
    def save_quantized_state_dict(self, save_path: Union[str, Path]):
        cur_state_dict = self.model.state_dict()
        for fqn, mod in self.model.named_modules():
            if isinstance(mod, torch.nn.Linear):
                int8_weight, scales, _ = dynamically_quantize_per_channel(mod.weight.float(), -128, 127, torch.int8)
                cur_state_dict[f"{fqn}.weight"] = int8_weight.to('cpu')
                cur_state_dict[f"{fqn}.scales"] = scales.to(mod.weight.dtype).to('cpu')
        torch.save(cur_state_dict, save_path)

    def convert_for_runtime(self):
        replace_linear_weight_only_int8_per_channel(self.model)
        return self.model
    

class WeightOnlyInt4QuantHelper:
    def __init__(self, model, groupsize=32, inner_k_tiles=8, padding_allowed=True):
        self.mod = model
        self.groupsize = groupsize
        self.inner_k_tiles = inner_k_tiles
        self.padding_allowed = padding_allowed
        assert groupsize in [32, 64, 128, 256]
        assert inner_k_tiles in [2, 4, 8]

    @torch.no_grad()
    def save_quantized_state_dict(self, save_path: Union[str, Path]):
        cur_state_dict = self.mod.state_dict()
        for fqn, mod in self.mod.named_modules():
            if isinstance(mod, torch.nn.Linear):
                assert not mod.bias
                out_features = mod.out_features
                in_features = mod.in_features
                assert out_features % 8 == 0, "require out_features % 8 == 0"
                print(f"linear: {fqn}, in={in_features}, out={out_features}")

                weight = mod.weight.data
                if not _check_linear_int4_k(in_features, self.groupsize, self.inner_k_tiles):
                    if self.padding_allowed:
                        import torch.nn.functional as F
                        print(f"warning: {fqn} is padded to satisfy in_features % 1024 == 0")
                        padded_in_features = find_multiple(in_features, 1024)
                        weight = F.pad(weight, pad=(0, padded_in_features - in_features))
                    else:
                        print(f"warning: {fqn} is skipped, int4 requires that in_features is 32, 64, or is divisible by 1024, " +
                            "and that groupsize and inner_k_tiles*16 evenly divide into it")
                        continue
                weight_int4pack, scales_and_zeros = prepare_int4_weight_and_scales_and_zeros(
                    weight.to(torch.bfloat16), self.groupsize, self.inner_k_tiles
                )
                cur_state_dict[f"{fqn}.weight"] = weight_int4pack.to('cpu')
                cur_state_dict[f"{fqn}.scales_and_zeros"] = scales_and_zeros.to('cpu')

        save_path = Path(save_path)
        if save_path.exists():
            save_path.unlink()
        torch.save(cur_state_dict, save_path)

    def convert_for_runtime(self, use_cuda):
        replace_linear_int4(self.mod, self.groupsize, self.inner_k_tiles, self.padding_allowed, use_cuda)
        return self.mod
    
    
def _check_linear_int4_k(k, groupsize = 1, inner_k_tiles = 1):
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
                    use_cuda=use_cuda
                )
                setattr(module, name, new_module)
        else:
            replace_linear_int4(child, groupsize, inner_k_tiles, padding_allowed, use_cuda)


def prepare_int4_weight_and_scales_and_zeros(weight_bf16, groupsize, inner_k_tiles):
    weight_int32, scales_and_zeros = group_quantize_tensor(
        weight_bf16, n_bit=4, groupsize=groupsize
    )
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
    return scales.to(torch.bfloat16).reshape(w.shape[0], -1), zeros.to(
        torch.bfloat16
    ).reshape(w.shape[0], -1)
    

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
    w_int32 = (
        to_quant.sub(min_val)
        .div(scales)
        .round()
        .clamp_(min_int, max_int)
        .to(torch.int32)
        .reshape_as(w)
    )
    return w_int32