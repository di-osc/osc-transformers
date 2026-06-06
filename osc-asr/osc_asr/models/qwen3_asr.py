from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from confection import Config

from osc_transformers import AutoRegressiveTransformer
from osc_transformers.sampler import SamplingParams
from osc_transformers.sequence import Sequence

from ..registry import Registry
from .base import ASRModel


class Qwen3ASRThinker(nn.Module):
    def __init__(
        self,
        text_model: AutoRegressiveTransformer,
        audio_tower: nn.Module,
        audio_token_id: int,
    ):
        super().__init__()
        self.text_model = text_model
        self.audio_tower = audio_tower
        self.audio_token_id = audio_token_id

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embedding

    def get_audio_features(
        self,
        input_features: torch.Tensor,
        feature_attention_mask: torch.Tensor | None = None,
        audio_feature_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
        if audio_feature_lengths is None:
            raise ValueError("audio_feature_lengths must be provided when feature_attention_mask is None")
        return self.audio_tower(input_features, feature_lens=audio_feature_lengths).last_hidden_state

    def get_placeholder_mask(self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor) -> torch.Tensor:
        audio_mask = input_ids == self.audio_token_id
        return audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)

    def prepare_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        input_features: torch.Tensor | None = None,
        feature_attention_mask: torch.Tensor | None = None,
        audio_feature_lengths: torch.Tensor | None = None,
        audio_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self.get_input_embeddings()(input_ids)
        if audio_features is None and input_features is not None:
            audio_features = self.get_audio_features(
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                audio_feature_lengths=audio_feature_lengths,
            )
        if audio_features is None:
            return inputs_embeds

        audio_mask = self.get_placeholder_mask(input_ids=input_ids, inputs_embeds=inputs_embeds)
        num_placeholders = int(audio_mask[..., 0].sum().item())
        if num_placeholders != audio_features.shape[0]:
            raise ValueError(
                f"Expected {num_placeholders} audio feature rows to match audio placeholders, "
                f"but got {audio_features.shape[0]}"
            )
        return inputs_embeds.masked_scatter(audio_mask, audio_features.to(inputs_embeds.device, inputs_embeds.dtype))

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        sampling_params: SamplingParams | list[SamplingParams] | None = None,
        attention_mask: torch.Tensor | None = None,
        input_features: torch.Tensor | None = None,
        feature_attention_mask: torch.Tensor | None = None,
        audio_feature_lengths: torch.Tensor | None = None,
        audio_features: torch.Tensor | None = None,
    ) -> list[list[int]]:
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        inputs_embeds = self.prepare_inputs_embeds(
            input_ids=input_ids,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            audio_feature_lengths=audio_feature_lengths,
            audio_features=audio_features,
        )
        if sampling_params is None:
            sampling_params = [SamplingParams(temperature=0.0, max_generate_tokens=512) for _ in range(input_ids.shape[0])]
        elif isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params for _ in range(input_ids.shape[0])]
        if len(sampling_params) != input_ids.shape[0]:
            raise ValueError("sampling_params must match batch size")
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")

        seqs = []
        for i in range(input_ids.shape[0]):
            if attention_mask is None:
                valid_token_ids = input_ids[i]
                valid_embeds = inputs_embeds[i]
            else:
                valid_mask = attention_mask[i].to(device=input_ids.device, dtype=torch.bool)
                valid_token_ids = input_ids[i][valid_mask]
                valid_embeds = inputs_embeds[i][valid_mask]
            seqs.append(
                Sequence(
                    token_ids=valid_token_ids.tolist(),
                    sampling_params=sampling_params[i],
                    prompt_embeds=valid_embeds,
                )
            )
        input_order = [seq.seq_id for seq in seqs]
        finished = {seq.seq_id: seq for seq in self.text_model.batch(seqs)}
        return [finished[seq_id].completion_token_ids for seq_id in input_order]


@dataclass(frozen=True)
class Qwen3ASRAudioEncoderConfig:
    num_mel_bins: int = 128
    encoder_layers: int = 32
    encoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    d_model: int = 1280
    dropout: float = 0.0
    attention_dropout: float = 0.0
    activation_function: str = "gelu"
    activation_dropout: float = 0.0
    scale_embedding: bool = False
    initializer_range: float = 0.02
    max_source_positions: int = 1500
    n_window: int = 100
    output_dim: int = 3584
    n_window_infer: int = 400
    conv_chunksize: int = 500
    downsample_hidden_size: int = 480

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "Qwen3ASRAudioEncoderConfig":
        field_names = cls.__dataclass_fields__.keys()
        return cls(**{name: values[name] for name in field_names if name in values})


@Registry.models.register("Qwen3ASRForConditionalGeneration")
class Qwen3ASRForConditionalGeneration(ASRModel):
    """Qwen3-ASR checkpoint adapter.

    The text decoder is represented as an osc-transformers
    AutoRegressiveTransformer config. The audio tower is tracked separately so
    its official weights can be mapped losslessly while the runtime grows ASR
    prefill support.
    """

    hf_architecture = "Qwen3ASRForConditionalGeneration"

    def __init__(
        self,
        checkpoint_dir: str | Path | None = None,
        *,
        config: dict[str, Any] | None = None,
        load_weights: bool = True,
    ):
        super().__init__(checkpoint_dir=checkpoint_dir, config=config)
        self.load_weights = load_weights
        if load_weights:
            if self.checkpoint_dir is None:
                raise ValueError("checkpoint_dir is required when load_weights=True")
            self.model = self.load()

    @property
    def thinker_config(self) -> dict[str, Any]:
        return self.hf_config["thinker_config"]

    @property
    def text_config(self) -> dict[str, Any]:
        return self.thinker_config["text_config"]

    @property
    def audio_config(self) -> Qwen3ASRAudioEncoderConfig:
        return Qwen3ASRAudioEncoderConfig.from_dict(self.thinker_config["audio_config"])

    @property
    def audio_token_id(self) -> int:
        return self.thinker_config["audio_token_id"]

    @property
    def support_languages(self) -> list[str]:
        return self.hf_config.get("support_languages", [])

    @property
    def text_osc_config(self) -> Config:
        text = self.text_config
        template = """
        [model]
        @architecture = "AutoRegressiveTransformer"
        num_layers = {num_hidden_layers}
        prenorm = "True"

        [model.attention]
        @attention = "PagedAttention"
        in_dim = {hidden_size}
        num_heads = {num_attention_heads}
        head_dim = {head_dim}
        num_query_groups = {num_key_value_heads}
        rope_base = {rope_theta}
        q_bias = "{attention_bias}"
        k_bias = "{attention_bias}"
        v_bias = "{attention_bias}"
        o_bias = "{attention_bias}"

        [model.attention.q_norm]
        @normalization = "RMSNorm"
        in_dim = {head_dim}
        eps = {rms_norm_eps}

        [model.attention.k_norm]
        @normalization = "RMSNorm"
        in_dim = {head_dim}
        eps = {rms_norm_eps}

        [model.embedding]
        @embedding = "VocabEmbedding"
        num_embeddings = {vocab_size}
        embedding_dim = {hidden_size}

        [model.feedforward]
        @feedforward = "SwiGLU"
        in_dim = {hidden_size}
        hidden_dim = {intermediate_size}
        up_bias = "False"
        gate_bias = "False"
        down_bias = "False"

        [model.head]
        @head = "LMHead"
        in_dim = {hidden_size}
        out_dim = {vocab_size}
        bias = "False"

        [model.norm]
        @normalization = "RMSNorm"
        in_dim = {hidden_size}
        eps = {rms_norm_eps}
        """
        values = {
            **text,
            "attention_bias": str(bool(text.get("attention_bias", False))),
        }
        return Config().from_str(template.format(**values))

    def build_text_model(self, empty_init: bool = True) -> AutoRegressiveTransformer:
        return AutoRegressiveTransformer.from_config(self.text_osc_config, empty_init=empty_init)

    def build_audio_tower(self):
        from .audio import Qwen3ASRAudioEncoder

        return Qwen3ASRAudioEncoder(self.audio_config)

    def build_model(self, empty_init: bool = True) -> Qwen3ASRThinker:
        return Qwen3ASRThinker(
            text_model=self.build_text_model(empty_init=empty_init),
            audio_tower=self.build_audio_tower(),
            audio_token_id=self.audio_token_id,
        )

    def setup(
        self,
        eos_id: int | list[int] | None = None,
        gpu_memory_utilization: float | None = None,
        device: str = "cuda",
        **kwargs,
    ) -> None:
        if not hasattr(self, "model"):
            raise ValueError("Model weights are not loaded; call load_asr_model(..., load_weights=True)")
        dtype = str_to_dtype(self.thinker_config.get("dtype", self.text_config.get("torch_dtype", "bfloat16")))
        self.device = torch.device(device)
        self.dtype = dtype
        self.model.audio_tower.to(device=device, dtype=dtype)
        self.model.text_model.setup(
            max_model_len=self.text_config.get("max_position_embeddings", 4096),
            gpu_memory_utilization=gpu_memory_utilization,
            eos=eos_id or self.hf_config.get("eos_token_id") or [151645, 151643],
            dtype=dtype,
            device=device,
            model_name=self.hf_architecture,
            **kwargs,
        )

    @torch.inference_mode()
    def generate(self, **kwargs) -> list[list[int]]:
        if not hasattr(self, "model"):
            raise ValueError("Model weights are not loaded; call load_asr_model(..., load_weights=True)")
        return self.model.generate(**kwargs)

    def load_checkpoint(self, model: Qwen3ASRThinker, states: dict[str, torch.Tensor]) -> Qwen3ASRThinker:
        model.load_state_dict(states, strict=True, assign=True)
        return model.eval()

    def resolve_missing_weights(self, states: dict[str, torch.Tensor], weight_map: dict[str, str]) -> None:
        hf_head_key = "thinker.lm_head.weight"
        local_head_key = "text_model.head.predictor.weight"
        if (
            self.text_config.get("tie_word_embeddings", False)
            and hf_head_key in weight_map
            and "text_model.embedding.embed.weight" in states
        ):
            states[local_head_key] = states["text_model.embedding.embed.weight"]
            weight_map.pop(hf_head_key)

    def load(self) -> Qwen3ASRThinker:
        model = self.build_model(empty_init=True)
        states = self.convert_checkpoint()
        return self.load_checkpoint(model=model, states=states)

    def get_placeholder_mask(self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor) -> torch.Tensor:
        audio_mask = input_ids == self.audio_token_id
        return audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)

    def merge_audio_features(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        audio_features: torch.Tensor,
    ) -> torch.Tensor:
        audio_mask = self.get_placeholder_mask(input_ids=input_ids, inputs_embeds=inputs_embeds)
        num_placeholders = int(audio_mask[..., 0].sum().item())
        if num_placeholders != audio_features.shape[0]:
            raise ValueError(
                f"Expected {num_placeholders} audio feature rows to match audio placeholders, "
                f"but got {audio_features.shape[0]}"
            )
        return inputs_embeds.masked_scatter(audio_mask, audio_features.to(inputs_embeds.device, inputs_embeds.dtype))

    @property
    def weight_map(self) -> dict[str, str]:
        text = self.text_config
        audio = self.audio_config
        weight_map = {
            "thinker.model.embed_tokens.weight": "text_model.embedding.embed.weight",
            "thinker.model.norm.weight": "text_model.head_norm.weight",
            "thinker.lm_head.weight": "text_model.head.predictor.weight",
            "thinker.audio_tower.conv2d1.weight": "audio_tower.conv2d1.weight",
            "thinker.audio_tower.conv2d1.bias": "audio_tower.conv2d1.bias",
            "thinker.audio_tower.conv2d2.weight": "audio_tower.conv2d2.weight",
            "thinker.audio_tower.conv2d2.bias": "audio_tower.conv2d2.bias",
            "thinker.audio_tower.conv2d3.weight": "audio_tower.conv2d3.weight",
            "thinker.audio_tower.conv2d3.bias": "audio_tower.conv2d3.bias",
            "thinker.audio_tower.conv_out.weight": "audio_tower.conv_out.weight",
            "thinker.audio_tower.ln_post.weight": "audio_tower.ln_post.weight",
            "thinker.audio_tower.ln_post.bias": "audio_tower.ln_post.bias",
            "thinker.audio_tower.proj1.weight": "audio_tower.proj1.weight",
            "thinker.audio_tower.proj1.bias": "audio_tower.proj1.bias",
            "thinker.audio_tower.proj2.weight": "audio_tower.proj2.weight",
            "thinker.audio_tower.proj2.bias": "audio_tower.proj2.bias",
        }

        for i in range(text["num_hidden_layers"]):
            weight_map[f"thinker.model.layers.{i}.input_layernorm.weight"] = (
                f"text_model.layers.{i}.attention_norm.weight"
            )
            weight_map[f"thinker.model.layers.{i}.post_attention_layernorm.weight"] = (
                f"text_model.layers.{i}.feedforward_norm.weight"
            )
            weight_map[f"thinker.model.layers.{i}.self_attn.q_proj.weight"] = (
                f"text_model.layers.{i}.attention.q_proj.weight"
            )
            weight_map[f"thinker.model.layers.{i}.self_attn.k_proj.weight"] = (
                f"text_model.layers.{i}.attention.k_proj.weight"
            )
            weight_map[f"thinker.model.layers.{i}.self_attn.v_proj.weight"] = (
                f"text_model.layers.{i}.attention.v_proj.weight"
            )
            weight_map[f"thinker.model.layers.{i}.self_attn.o_proj.weight"] = (
                f"text_model.layers.{i}.attention.o_proj.weight"
            )
            weight_map[f"thinker.model.layers.{i}.self_attn.q_norm.weight"] = (
                f"text_model.layers.{i}.attention.q_norm.weight"
            )
            weight_map[f"thinker.model.layers.{i}.self_attn.k_norm.weight"] = (
                f"text_model.layers.{i}.attention.k_norm.weight"
            )
            weight_map[f"thinker.model.layers.{i}.mlp.gate_proj.weight"] = (
                f"text_model.layers.{i}.feedforward.gate_proj.weight"
            )
            weight_map[f"thinker.model.layers.{i}.mlp.up_proj.weight"] = (
                f"text_model.layers.{i}.feedforward.up_proj.weight"
            )
            weight_map[f"thinker.model.layers.{i}.mlp.down_proj.weight"] = (
                f"text_model.layers.{i}.feedforward.down_proj.weight"
            )

        for i in range(audio.encoder_layers):
            prefix = f"thinker.audio_tower.layers.{i}"
            local = f"audio_tower.layers.{i}"
            for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
                weight_map[f"{prefix}.self_attn.{proj}.weight"] = f"{local}.self_attn.{proj}.weight"
                weight_map[f"{prefix}.self_attn.{proj}.bias"] = f"{local}.self_attn.{proj}.bias"
            for norm in ("self_attn_layer_norm", "final_layer_norm"):
                weight_map[f"{prefix}.{norm}.weight"] = f"{local}.{norm}.weight"
                weight_map[f"{prefix}.{norm}.bias"] = f"{local}.{norm}.bias"
            for fc in ("fc1", "fc2"):
                weight_map[f"{prefix}.{fc}.weight"] = f"{local}.{fc}.weight"
                weight_map[f"{prefix}.{fc}.bias"] = f"{local}.{fc}.bias"

        return weight_map


def str_to_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32
    if dtype == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype: {dtype}")
