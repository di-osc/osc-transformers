import math
import pickle
import sys
from io import BytesIO
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
import torch.utils._device
from torch.serialization import normalize_storage_type
from typing import Dict


def find_multiple(n: int, k: int) -> int:
    assert k > 0
    if n % k == 0:
        return n
    return n + k - (n % k)


def num_parameters(module: nn.Module, requires_grad: Optional[bool] = None) -> int:
    total = 0
    for p in module.parameters():
        if requires_grad is None or p.requires_grad == requires_grad:
            if hasattr(p, "quant_state"):
                # bitsandbytes 4bit layer support
                total += math.prod(p.quant_state[1])
            else:
                total += p.numel()
    return total


def check_valid_checkpoint_dir(checkpoint_dir: Path) -> None:
    files = {
        "lit_model.pth": (checkpoint_dir / "lit_model.pth").is_file(),
        "lit_config.json": (checkpoint_dir / "lit_config.json").is_file(),
        "tokenizer.json OR tokenizer.model": (checkpoint_dir / "tokenizer.json").is_file() or (
            checkpoint_dir / "tokenizer.model"
        ).is_file(),
        "tokenizer_config.json": (checkpoint_dir / "tokenizer_config.json").is_file(),
    }
    if checkpoint_dir.is_dir():
        if all(files.values()):
            # we're good
            return
        problem = f" is missing the files: {[f for f, exists in files.items() if not exists]!r}"
    else:
        problem = " is not a checkpoint directory"

    # list locally available checkpoints
    available = list(Path("checkpoints").glob("*/*"))
    if available:
        options = "\n --checkpoint_dir ".join([""] + [repr(str(p.resolve())) for p in available])
        extra = f"\nYou have downloaded locally:{options}\n"
    else:
        extra = ""

    error_message = (
        f"--checkpoint_dir {str(checkpoint_dir.absolute())!r}{problem}."
        "\nFind download instructions at https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials\n"
        f"{extra}\nSee all download options by running:\n python scripts/download.py"
    )
    print(error_message, file=sys.stderr)
    raise SystemExit(1)


class SavingProxyForStorage:
    def __init__(self, obj, saver, protocol_version=5):
        self.protocol_version = protocol_version
        self.saver = saver
        if not (isinstance(obj, torch.storage.TypedStorage) or torch.is_storage(obj)):
            raise TypeError(f"expected storage, not {type(obj)}")

        # this logic is taken from PyTorch 2.0+ torch/serialization.py
        if isinstance(obj, torch.storage.TypedStorage):
            # PT upstream wants to deprecate this eventually...
            storage = obj._untyped_storage
            storage_type_str = obj._pickle_storage_type()
            storage_type = getattr(torch, storage_type_str)
            storage_numel = obj._size()
        else:
            storage = obj
            storage_type = normalize_storage_type(type(obj))
            storage_numel = storage.nbytes()

        storage_key = saver._write_storage_and_return_key(storage)
        location = torch.serialization.location_tag(storage)

        self.storage_info = ("storage", storage_type, storage_key, location, storage_numel)

    def __reduce_ex__(self, protocol_version):
        assert False, "this should be handled with out of band"


class SavingProxyForTensor:
    def __init__(self, tensor, saver, protocol_version=5):
        self.protocol_version = protocol_version
        self.reduce_ret_fn, reduce_args = tensor.__reduce_ex__(protocol_version)
        if reduce_args[0] == torch._utils._rebuild_tensor_v2:
            # for Tensors with Python attributes
            (a0, a1, (storage, *a2_other), *other_reduce_args) = reduce_args
            assert isinstance(storage, torch.storage.TypedStorage), "Please check for updates"
            storage_proxy = SavingProxyForStorage(storage, saver, protocol_version=protocol_version)
            self.reduce_args = (a0, a1, (storage_proxy, *a2_other), *other_reduce_args)
        else:
            (storage, *other_reduce_args) = reduce_args
            assert isinstance(storage, torch.storage.TypedStorage), "Please check for updates"
            storage_proxy = SavingProxyForStorage(storage, saver, protocol_version=protocol_version)
            self.reduce_args = (storage_proxy, *other_reduce_args)

    def __reduce_ex__(self, protocol_version):
        if protocol_version != self.protocol_version:
            raise RuntimeError(f"Unexpected protocol version: expected {self.protocol_version}, got {protocol_version}")
        return self.reduce_ret_fn, self.reduce_args


class IncrementalPyTorchPickler(pickle.Pickler):
    def __init__(self, saver, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.storage_dtypes = {}
        self.saver = saver
        self.id_map = {}

    # this logic is taken from PyTorch 2.0+ torch/serialization.py
    def persistent_id(self, obj):
        # FIXME: the docs say that persistent_id should only return a string
        # but torch store returns tuples. This works only in the binary protocol
        # see
        # https://docs.python.org/2/library/pickle.html#pickling-and-unpickling-external-objects
        # https://github.com/python/cpython/blob/master/Lib/pickle.py#L527-L537
        if isinstance(obj, SavingProxyForStorage):
            return obj.storage_info

        if isinstance(obj, torch.storage.TypedStorage) or torch.is_storage(obj):
            if isinstance(obj, torch.storage.TypedStorage):
                # TODO: Once we decide to break serialization FC, this case
                # can be deleted
                storage = obj._untyped_storage
                storage_dtype = obj.dtype
                storage_type_str = obj._pickle_storage_type()
                storage_type = getattr(torch, storage_type_str)
                storage_numel = obj._size()

            else:
                storage = obj
                storage_dtype = torch.uint8
                storage_type = normalize_storage_type(type(obj))
                storage_numel = storage.nbytes()

            # If storage is allocated, ensure that any other saved storages
            # pointing to the same data all have the same dtype. If storage is
            # not allocated, don't perform this check
            if storage.data_ptr() != 0:
                if storage.data_ptr() in self.storage_dtypes:
                    if storage_dtype != self.storage_dtypes[storage.data_ptr()]:
                        raise RuntimeError(
                            "Cannot save multiple tensors or storages that view the same data as different types"
                        )
                else:
                    self.storage_dtypes[storage.data_ptr()] = storage_dtype

            storage_key = self.id_map.get(storage._cdata)
            if storage_key is None:
                storage_key = self.saver._write_storage_and_return_key(storage)
                self.id_map[storage._cdata] = storage_key
            location = torch.serialization.location_tag(storage)

            return ("storage", storage_type, storage_key, location, storage_numel)

        return None



class IncrementalSaver:
    def __init__(self, name):
        self.name = name
        self.zipfile = torch._C.PyTorchFileWriter(str(name))
        self.has_saved = False
        self.next_key = 0

    def __enter__(self):
        return self

    def store_early(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return SavingProxyForTensor(tensor, self)
        raise TypeError(f"can only store tensors early, not {type(tensor)}")

    def save(self, obj):
        if self.has_saved:
            raise RuntimeError("have already saved")
        # Write the pickle data for `obj`
        data_buf = BytesIO()
        pickler = IncrementalPyTorchPickler(self, data_buf, protocol=5)
        pickler.dump(obj)
        data_value = data_buf.getvalue()
        self.zipfile.write_record("data.pkl", data_value, len(data_value))
        self.has_saved = True

    def _write_storage_and_return_key(self, storage):
        if self.has_saved:
            raise RuntimeError("have already saved")
        key = self.next_key
        self.next_key += 1
        name = f"data/{key}"
        if storage.device.type != "cpu":
            storage = storage.cpu()
        num_bytes = storage.nbytes()
        self.zipfile.write_record(name, storage.data_ptr(), num_bytes)
        return key

    def __exit__(self, type, value, traceback):
        self.zipfile.write_end_of_file()
        

class WeightMap:
    def __init__(self) -> None:
        pass
    
    @classmethod
    def get_llama2_weight_map(self, num_blocks: int = 32) -> Dict:
        """获取llama2 7b的权重映射表
        """
        weight_map = {
        "model.embed_tokens.weight": "embedding.embed.weight",
        "model.norm.weight": "head_norm.weight",
        "lm_head.weight": "head.predictor.weight",
        }
        
        for i in range(num_blocks):
            weight_map[f"model.layers.{i}.input_layernorm.weight"] = f"blocks.{i}.attention_norm.weight"
            weight_map[f"model.layers.{i}.post_attention_layernorm.weight"] = f"blocks.{i}.feedforward_norm.weight"
            weight_map[f"model.layers.{i}.self_attn.q_proj.weight"] = f"blocks.{i}.attention.q_proj.weight"
            weight_map[f"model.layers.{i}.self_attn.k_proj.weight"] = f"blocks.{i}.attention.k_proj.weight"
            weight_map[f"model.layers.{i}.self_attn.v_proj.weight"] = f"blocks.{i}.attention.v_proj.weight"
            weight_map[f"model.layers.{i}.self_attn.o_proj.weight"] = f"blocks.{i}.attention.o_proj.weight"
            weight_map[f"model.layers.{i}.mlp.gate_proj.weight"] = f"blocks.{i}.feedforward.gate_proj.weight"
            weight_map[f"model.layers.{i}.mlp.up_proj.weight"] = f"blocks.{i}.feedforward.up_proj.weight"
            weight_map[f"model.layers.{i}.mlp.down_proj.weight"] = f"blocks.{i}.feedforward.down_proj.weight"
            
        return weight_map
        