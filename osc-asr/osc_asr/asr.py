from pathlib import Path
from typing import Any

import numpy as np
import torch

from osc_transformers import SamplingParams

from .models import Qwen3ASRForConditionalGeneration, load_asr_model


class ASR:
    def __init__(
        self,
        checkpoint_dir: str | Path,
        processor: Any | None = None,
        model: Qwen3ASRForConditionalGeneration | None = None,
        gpu_memory_utilization: float | None = None,
        device: str = "cuda",
        cuda_graph: bool = True,
        setup_model: bool = True,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.processor = processor or self._load_processor(checkpoint_dir)
        self.model = model or load_asr_model(checkpoint_dir)
        if setup_model and hasattr(self.model, "setup"):
            self.model.setup(
                gpu_memory_utilization=gpu_memory_utilization,
                device=device,
                cuda_graph=cuda_graph,
            )

    def transcribe(
        self,
        audio,
        sampling_rate: int | None = None,
        context: str = "",
        language: str | None = None,
        sampling_params: SamplingParams | None = None,
        **processor_kwargs,
    ) -> str:
        model_inputs = self.prepare_inputs(
            audio=audio,
            sampling_rate=sampling_rate,
            context=context,
            language=language,
            **processor_kwargs,
        )
        model_inputs = self._move_inputs_to_model(model_inputs)
        token_ids = self.model.generate(sampling_params=sampling_params, **model_inputs)
        return self._parse_asr_output(self.decode(token_ids)[0], user_language=language)[1]

    def prepare_inputs(
        self,
        audio,
        sampling_rate: int | None = None,
        context: str = "",
        language: str | None = None,
        **kwargs,
    ) -> dict:
        audio = normalize_audio_input(audio, sampling_rate=sampling_rate)
        processor_kwargs = {
            "text": [self._build_text_prompt(context=context, language=language)],
            "audio": [audio],
            "return_tensors": "pt",
            "padding": True,
            "sampling_rate": 16000,
            **kwargs,
        }
        return self.processor(**processor_kwargs)

    def decode(self, batch_token_ids: list[list[int]]) -> list[str]:
        if hasattr(self.processor, "batch_decode"):
            return self.processor.batch_decode(
                batch_token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        if hasattr(self.processor, "tokenizer") and hasattr(self.processor.tokenizer, "batch_decode"):
            return self.processor.tokenizer.batch_decode(
                batch_token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        raise AttributeError("processor must provide batch_decode or tokenizer.batch_decode")

    def _build_text_prompt(self, context: str = "", language: str | None = None) -> str:
        messages = [
            {"role": "system", "content": context or ""},
            {"role": "user", "content": [{"type": "audio", "audio": ""}]},
        ]
        if not hasattr(self.processor, "apply_chat_template"):
            raise AttributeError("processor must provide apply_chat_template for Qwen3-ASR prompts")
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        if language:
            prompt = prompt + f"language {language}<asr_text>"
        return prompt

    def _move_inputs_to_model(self, model_inputs):
        device = self._model_device()
        dtype = self._model_dtype()
        if hasattr(model_inputs, "to"):
            moved = model_inputs.to(device)
            if dtype is not None:
                moved = moved.to(dtype)
            return moved
        return _move_value(model_inputs, device=device, dtype=dtype)

    def _model_device(self) -> torch.device | str:
        if hasattr(self.model, "device"):
            return self.model.device
        if hasattr(self.model, "model"):
            try:
                return next(self.model.model.parameters()).device
            except StopIteration:
                pass
        return torch.device("cpu")

    def _model_dtype(self) -> torch.dtype | None:
        if hasattr(self.model, "dtype"):
            return self.model.dtype
        if hasattr(self.model, "model"):
            try:
                return next(self.model.model.parameters()).dtype
            except StopIteration:
                pass
        return None

    @staticmethod
    def _parse_asr_output(raw: str, user_language: str | None = None) -> tuple[str, str]:
        text = str(raw or "").strip()
        if not text:
            return "", ""
        if user_language:
            return user_language, text
        tag = "<asr_text>"
        if tag not in text:
            return "", text
        meta, content = text.split(tag, 1)
        if "language none" in meta.lower() and not content.strip():
            return "", ""
        language = ""
        marker = "language "
        lower_meta = meta.lower()
        if marker in lower_meta:
            start = lower_meta.rfind(marker) + len(marker)
            language = meta[start:].strip()
        return language, content.strip()

    @staticmethod
    def _load_processor(checkpoint_dir: str | Path):
        try:
            from transformers import AutoConfig, AutoProcessor
        except Exception as exc:
            raise ImportError("Please install transformers to load the Qwen3-ASR processor") from exc
        from .processing_qwen3_asr import Qwen3ASRConfig, Qwen3ASRProcessor

        _register_auto_config(AutoConfig, "qwen3_asr", Qwen3ASRConfig)
        _register_auto_processor(AutoProcessor, Qwen3ASRConfig, Qwen3ASRProcessor)
        try:
            from qwen_asr.core.transformers_backend import Qwen3ASRConfig as OfficialQwen3ASRConfig
            from qwen_asr.core.transformers_backend import Qwen3ASRProcessor as OfficialQwen3ASRProcessor

            _register_auto_config(AutoConfig, "qwen3_asr", OfficialQwen3ASRConfig)
            _register_auto_processor(AutoProcessor, OfficialQwen3ASRConfig, OfficialQwen3ASRProcessor)
        except Exception:
            pass
        return AutoProcessor.from_pretrained(checkpoint_dir, trust_remote_code=True)


def _register_auto_config(auto_config, model_type: str, config_cls) -> None:
    try:
        auto_config.register(model_type, config_cls, exist_ok=True)
    except TypeError:
        try:
            auto_config.register(model_type, config_cls)
        except ValueError:
            pass


def _register_auto_processor(auto_processor, config_cls, processor_cls) -> None:
    try:
        auto_processor.register(config_cls, processor_cls, exist_ok=True)
    except TypeError:
        try:
            auto_processor.register(config_cls, processor_cls)
        except ValueError:
            pass


def _move_value(value, device: torch.device | str, dtype: torch.dtype | None = None):
    if isinstance(value, torch.Tensor):
        if dtype is not None and value.is_floating_point():
            return value.to(device=device, dtype=dtype)
        return value.to(device=device)
    if isinstance(value, dict):
        return value.__class__({key: _move_value(item, device=device, dtype=dtype) for key, item in value.items()})
    if isinstance(value, list):
        return [_move_value(item, device=device, dtype=dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_value(item, device=device, dtype=dtype) for item in value)
    return value


def normalize_audio_input(audio, sampling_rate: int | None = None) -> np.ndarray:
    if isinstance(audio, str):
        import librosa

        waveform, sr = librosa.load(audio, sr=None, mono=False)
    elif isinstance(audio, tuple) and len(audio) == 2:
        waveform, sr = audio
    else:
        if sampling_rate is None:
            sampling_rate = 16000
        waveform, sr = audio, sampling_rate

    waveform = to_mono(np.asarray(waveform))
    sr = int(sr)
    if sr != 16000:
        import librosa

        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
    return float_range_normalize(waveform)


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32)
    if audio.ndim == 2:
        if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
            audio = audio.T
        return np.mean(audio, axis=-1).astype(np.float32)
    raise ValueError(f"Unsupported audio ndim={audio.ndim}")


def float_range_normalize(audio: np.ndarray) -> np.ndarray:
    audio = audio.astype(np.float32)
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak == 0.0:
        return audio
    if peak > 1.0:
        audio = audio / peak
    return np.clip(audio, -1.0, 1.0).astype(np.float32)
