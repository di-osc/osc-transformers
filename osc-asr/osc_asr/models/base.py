import json
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from ..registry import Registry


class ASRModel:
    """Base class for ASR models loaded from Hugging Face checkpoints."""

    hf_architecture: str

    def __init__(self, checkpoint_dir: str | Path | None = None, *, config: dict[str, Any] | None = None):
        if checkpoint_dir is None and config is None:
            raise ValueError("Either checkpoint_dir or config must be provided")
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.hf_config = config or self._load_config(self.checkpoint_dir)
        architectures = self.hf_config.get("architectures", [])
        if self.hf_architecture not in architectures:
            raise ValueError(
                f"Only support {self.hf_architecture} model, current model architectures are: {architectures}"
            )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ASRModel":
        return cls(config=config, load_weights=False)

    @staticmethod
    def _load_config(checkpoint_dir: Path) -> dict[str, Any]:
        config_path = checkpoint_dir / "config.json"
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)

    @property
    def weight_map(self) -> dict[str, str]:
        raise NotImplementedError("Method not implemented")

    def convert_checkpoint(self) -> dict[str, torch.Tensor]:
        pytorch_model = self.checkpoint_dir / "pytorch_model.bin"
        pytorch_idx_file = self.checkpoint_dir / "pytorch_model.bin.index.json"
        safetensors_model = self.checkpoint_dir / "model.safetensors"
        safetensors_idx_file = self.checkpoint_dir / "model.safetensors.index.json"
        if pytorch_model.exists() or pytorch_idx_file.exists():
            return self.convert_pytorch_format()
        if safetensors_model.exists() or safetensors_idx_file.exists():
            return self.convert_safetensor_format()
        raise FileNotFoundError("No pytorch_model.bin or model.safetensors found")

    def convert_pytorch_format(self) -> dict[str, torch.Tensor]:
        states = {}
        weight_map = self.weight_map
        index_file = self.checkpoint_dir / "pytorch_model.bin.index.json"
        if index_file.exists():
            with open(index_file, encoding="utf-8") as f:
                index = json.load(f)
            files = [self.checkpoint_dir / file for file in sorted(set(index["weight_map"].values()))]
        else:
            files = [self.checkpoint_dir / "pytorch_model.bin"]

        for file in files:
            weights = torch.load(str(file), map_location="cpu", weights_only=True, mmap=True)
            for key, tensor in weights.items():
                if key not in weight_map:
                    logger.warning(f"{key} not in weight_map")
                    continue
                states[weight_map.pop(key)] = tensor
        self.resolve_missing_weights(states, weight_map)
        if weight_map:
            raise AssertionError(f"Some weights are not in the weight map: {weight_map}")
        return states

    def convert_safetensor_format(self) -> dict[str, torch.Tensor]:
        states = {}
        weight_map = self.weight_map
        index_file = self.checkpoint_dir / "model.safetensors.index.json"
        if index_file.exists():
            with open(index_file, encoding="utf-8") as f:
                index = json.load(f)
            files = [self.checkpoint_dir / file for file in sorted(set(index["weight_map"].values()))]
        else:
            files = [self.checkpoint_dir / "model.safetensors"]
        try:
            from safetensors import safe_open
        except Exception as exc:
            raise ImportError("Please install safetensors first, run `pip install safetensors`") from exc

        for file in files:
            with safe_open(file, framework="pt") as f:
                for key in f.keys():
                    if key not in weight_map:
                        logger.warning(f"{key} not in weight_map")
                        continue
                    states[weight_map.pop(key)] = f.get_tensor(key)
        self.resolve_missing_weights(states, weight_map)
        if weight_map:
            raise AssertionError(f"Some weights are not in the weight map: {weight_map}")
        return states

    def resolve_missing_weights(self, states: dict[str, torch.Tensor], weight_map: dict[str, str]) -> None:
        return None


def get_supported_hf_models() -> list[str]:
    return list(Registry.models.get_all())


def load_asr_model(checkpoint_dir: str | Path, *, load_weights: bool = True) -> ASRModel:
    checkpoint_dir = Path(checkpoint_dir)
    config = ASRModel._load_config(checkpoint_dir)
    model_name = config["architectures"][0]
    allowed_models = get_supported_hf_models()
    if model_name not in allowed_models:
        logger.error(f"Model {model_name} is not supported. Supported models are: {allowed_models}")
        raise ValueError(f"Unsupported ASR model: {model_name}")
    model_cls = Registry.models.get(model_name)
    return model_cls(checkpoint_dir=checkpoint_dir, config=config, load_weights=load_weights)
