import json

import numpy as np
import pytest
import torch
from confection import Config
from osc_asr import ASR
from osc_asr.models import Qwen3ASRForConditionalGeneration, load_asr_model
from osc_asr.models.audio import Qwen3ASRAudioEncoder, SinusoidsPositionEmbedding, get_feat_extract_output_lengths
from osc_asr.models.qwen3_asr import Qwen3ASRThinker

from osc_transformers import AutoRegressiveTransformer
from osc_transformers.embedding.vocab import VocabEmbedding
from osc_transformers.sequence import Sequence


def _qwen3_asr_config(num_text_layers: int = 2, num_audio_layers: int = 2) -> dict:
    return {
        "architectures": ["Qwen3ASRForConditionalGeneration"],
        "model_type": "qwen3_asr",
        "support_languages": ["Chinese", "English"],
        "thinker_config": {
            "model_type": "qwen3_asr_thinker",
            "audio_token_id": 151676,
            "audio_start_token_id": 151669,
            "audio_end_token_id": 151670,
            "dtype": "bfloat16",
            "audio_config": {
                "model_type": "qwen3_asr_audio_encoder",
                "num_mel_bins": 128,
                "encoder_layers": num_audio_layers,
                "encoder_attention_heads": 2,
                "encoder_ffn_dim": 32,
                "d_model": 16,
                "dropout": 0,
                "attention_dropout": 0,
                "activation_function": "gelu",
                "activation_dropout": 0,
                "scale_embedding": False,
                "initializer_range": 0.02,
                "max_source_positions": 1500,
                "n_window": 50,
                "output_dim": 16,
                "n_window_infer": 800,
                "conv_chunksize": 500,
                "downsample_hidden_size": 8,
            },
            "text_config": {
                "model_type": "qwen3_asr_text",
                "vocab_size": 151680,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": num_text_layers,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "hidden_act": "silu",
                "max_position_embeddings": 256,
                "initializer_range": 0.02,
                "rms_norm_eps": 1e-6,
                "use_cache": True,
                "tie_word_embeddings": True,
                "rope_theta": 1000000,
                "rope_scaling": {
                    "interleaved": True,
                    "mrope_interleaved": True,
                    "mrope_section": [2, 1, 1],
                    "rope_type": "default",
                    "type": "default",
                },
                "attention_bias": False,
                "attention_dropout": 0.0,
            },
        },
    }


def _small_qwen3_asr_config(num_text_layers: int = 0, num_audio_layers: int = 0) -> dict:
    cfg = _qwen3_asr_config(num_text_layers=num_text_layers, num_audio_layers=num_audio_layers)
    cfg["thinker_config"]["audio_token_id"] = 15
    cfg["thinker_config"]["text_config"]["vocab_size"] = 16
    return cfg


def test_loads_qwen3_asr_config_from_checkpoint_dir(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_qwen3_asr_config()), encoding="utf-8")

    model = load_asr_model(tmp_path, load_weights=False)

    assert isinstance(model, Qwen3ASRForConditionalGeneration)
    assert model.hf_architecture == "Qwen3ASRForConditionalGeneration"
    assert model.audio_token_id == 151676
    assert model.support_languages == ["Chinese", "English"]


def test_loads_pytorch_checkpoint_into_qwen3_asr_thinker(tmp_path):
    config = _small_qwen3_asr_config(num_text_layers=0, num_audio_layers=0)
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    source_weight = torch.randn(16, 16)
    torch.save(
        {
            "thinker.model.embed_tokens.weight": source_weight,
            "thinker.model.norm.weight": torch.ones(16),
            "thinker.lm_head.weight": torch.randn(16, 16),
            "thinker.audio_tower.conv2d1.weight": torch.randn(8, 1, 3, 3),
            "thinker.audio_tower.conv2d1.bias": torch.randn(8),
            "thinker.audio_tower.conv2d2.weight": torch.randn(8, 8, 3, 3),
            "thinker.audio_tower.conv2d2.bias": torch.randn(8),
            "thinker.audio_tower.conv2d3.weight": torch.randn(8, 8, 3, 3),
            "thinker.audio_tower.conv2d3.bias": torch.randn(8),
            "thinker.audio_tower.conv_out.weight": torch.randn(16, 128),
            "thinker.audio_tower.ln_post.weight": torch.ones(16),
            "thinker.audio_tower.ln_post.bias": torch.zeros(16),
            "thinker.audio_tower.proj1.weight": torch.randn(16, 16),
            "thinker.audio_tower.proj1.bias": torch.randn(16),
            "thinker.audio_tower.proj2.weight": torch.randn(16, 16),
            "thinker.audio_tower.proj2.bias": torch.randn(16),
        },
        tmp_path / "pytorch_model.bin",
    )

    model = load_asr_model(tmp_path, load_weights=True)

    torch.testing.assert_close(model.model.text_model.embedding.embed.weight, source_weight)


def test_loads_tied_embedding_checkpoint_without_lm_head(tmp_path):
    config = _small_qwen3_asr_config(num_text_layers=0, num_audio_layers=0)
    config["thinker_config"]["text_config"]["tie_word_embeddings"] = True
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    source_weight = torch.randn(16, 16)
    torch.save(
        {
            "thinker.model.embed_tokens.weight": source_weight,
            "thinker.model.norm.weight": torch.ones(16),
            "thinker.audio_tower.conv2d1.weight": torch.randn(8, 1, 3, 3),
            "thinker.audio_tower.conv2d1.bias": torch.randn(8),
            "thinker.audio_tower.conv2d2.weight": torch.randn(8, 8, 3, 3),
            "thinker.audio_tower.conv2d2.bias": torch.randn(8),
            "thinker.audio_tower.conv2d3.weight": torch.randn(8, 8, 3, 3),
            "thinker.audio_tower.conv2d3.bias": torch.randn(8),
            "thinker.audio_tower.conv_out.weight": torch.randn(16, 128),
            "thinker.audio_tower.ln_post.weight": torch.ones(16),
            "thinker.audio_tower.ln_post.bias": torch.zeros(16),
            "thinker.audio_tower.proj1.weight": torch.randn(16, 16),
            "thinker.audio_tower.proj1.bias": torch.randn(16),
            "thinker.audio_tower.proj2.weight": torch.randn(16, 16),
            "thinker.audio_tower.proj2.bias": torch.randn(16),
        },
        tmp_path / "pytorch_model.bin",
    )

    model = load_asr_model(tmp_path, load_weights=True)

    torch.testing.assert_close(model.model.text_model.head.predictor.weight, source_weight)


def test_loads_single_safetensors_checkpoint_like_official_qwen3_asr(tmp_path):
    from safetensors.torch import save_file

    config = _small_qwen3_asr_config(num_text_layers=0, num_audio_layers=0)
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    source_weight = torch.randn(16, 16)
    save_file(
        {
            "thinker.model.embed_tokens.weight": source_weight,
            "thinker.model.norm.weight": torch.ones(16),
            "thinker.lm_head.weight": torch.randn(16, 16),
            "thinker.audio_tower.conv2d1.weight": torch.randn(8, 1, 3, 3),
            "thinker.audio_tower.conv2d1.bias": torch.randn(8),
            "thinker.audio_tower.conv2d2.weight": torch.randn(8, 8, 3, 3),
            "thinker.audio_tower.conv2d2.bias": torch.randn(8),
            "thinker.audio_tower.conv2d3.weight": torch.randn(8, 8, 3, 3),
            "thinker.audio_tower.conv2d3.bias": torch.randn(8),
            "thinker.audio_tower.conv_out.weight": torch.randn(16, 128),
            "thinker.audio_tower.ln_post.weight": torch.ones(16),
            "thinker.audio_tower.ln_post.bias": torch.zeros(16),
            "thinker.audio_tower.proj1.weight": torch.randn(16, 16),
            "thinker.audio_tower.proj1.bias": torch.randn(16),
            "thinker.audio_tower.proj2.weight": torch.randn(16, 16),
            "thinker.audio_tower.proj2.bias": torch.randn(16),
        },
        tmp_path / "model.safetensors",
    )

    model = load_asr_model(tmp_path, load_weights=True)

    torch.testing.assert_close(model.model.text_model.embedding.embed.weight, source_weight)


def test_text_osc_config_uses_autoregressive_transformer():
    model = Qwen3ASRForConditionalGeneration.from_config(_qwen3_asr_config(num_text_layers=3))

    config = model.text_osc_config

    assert isinstance(config, Config)
    resolved = config.interpolate()["model"]
    assert resolved["@architecture"] == "AutoRegressiveTransformer"
    assert resolved["num_layers"] == 3
    assert resolved["attention"]["@attention"] == "PagedAttention"
    assert resolved["attention"]["num_heads"] == 2
    assert resolved["attention"]["num_query_groups"] == 1
    assert resolved["attention"]["rope_base"] == 1000000
    assert resolved["embedding"]["num_embeddings"] == 151680
    assert resolved["head"]["out_dim"] == 151680


def test_audio_config_preserves_official_encoder_shape_fields():
    model = Qwen3ASRForConditionalGeneration.from_config(_qwen3_asr_config(num_audio_layers=4))

    audio = model.audio_config

    assert audio.encoder_layers == 4
    assert audio.d_model == 16
    assert audio.encoder_attention_heads == 2
    assert audio.encoder_ffn_dim == 32
    assert audio.num_mel_bins == 128
    assert audio.output_dim == 16
    assert audio.n_window == 50
    assert audio.n_window_infer == 800
    assert audio.conv_chunksize == 500
    assert audio.downsample_hidden_size == 8


def test_weight_map_covers_text_and_audio_tower_keys():
    model = Qwen3ASRForConditionalGeneration.from_config(
        _qwen3_asr_config(num_text_layers=2, num_audio_layers=2)
    )

    weight_map = model.weight_map

    assert weight_map["thinker.model.embed_tokens.weight"] == "text_model.embedding.embed.weight"
    assert weight_map["thinker.model.norm.weight"] == "text_model.head_norm.weight"
    assert weight_map["thinker.lm_head.weight"] == "text_model.head.predictor.weight"
    assert (
        weight_map["thinker.model.layers.1.self_attn.q_proj.weight"]
        == "text_model.layers.1.attention.q_proj.weight"
    )
    assert (
        weight_map["thinker.model.layers.1.self_attn.q_norm.weight"]
        == "text_model.layers.1.attention.q_norm.weight"
    )
    assert (
        weight_map["thinker.model.layers.1.mlp.gate_proj.weight"]
        == "text_model.layers.1.feedforward.gate_proj.weight"
    )
    assert weight_map["thinker.audio_tower.conv2d1.weight"] == "audio_tower.conv2d1.weight"
    assert (
        weight_map["thinker.audio_tower.layers.1.self_attn.out_proj.bias"]
        == "audio_tower.layers.1.self_attn.out_proj.bias"
    )
    assert weight_map["thinker.audio_tower.proj2.weight"] == "audio_tower.proj2.weight"


def test_audio_feature_length_formula_matches_official_qwen3_asr():
    lengths = torch.tensor([1, 2, 3, 99, 100, 101, 199, 200, 201])

    actual = get_feat_extract_output_lengths(lengths)

    assert actual.tolist() == [1, 1, 1, 13, 13, 14, 26, 26, 27]


def test_sinusoid_position_embedding_starts_with_zero_sin_and_one_cos():
    embedding = SinusoidsPositionEmbedding(length=4, channels=6)

    values = embedding(3)

    assert values.shape == (3, 6)
    torch.testing.assert_close(values[0, :3], torch.zeros(3))
    torch.testing.assert_close(values[0, 3:], torch.ones(3))


def test_audio_encoder_outputs_one_vector_per_downsampled_frame():
    cfg = Qwen3ASRForConditionalGeneration.from_config(
        _qwen3_asr_config(num_audio_layers=1)
    ).audio_config
    cfg = cfg.__class__(
        num_mel_bins=8,
        encoder_layers=1,
        encoder_attention_heads=2,
        encoder_ffn_dim=32,
        d_model=16,
        output_dim=12,
        n_window=4,
        n_window_infer=8,
        downsample_hidden_size=4,
    )
    encoder = Qwen3ASRAudioEncoder(cfg).eval()
    feature_lens = torch.tensor([16, 8], dtype=torch.long)
    input_features = torch.randn(cfg.num_mel_bins, feature_lens.sum().item())

    with torch.no_grad():
        outputs = encoder(input_features, feature_lens=feature_lens)

    expected_frames = get_feat_extract_output_lengths(feature_lens).sum().item()
    assert outputs.last_hidden_state.shape == (expected_frames, cfg.output_dim)


def test_build_text_model_uses_osc_transformers_architecture():
    model = Qwen3ASRForConditionalGeneration.from_config(_qwen3_asr_config(num_text_layers=0))

    text_model = model.build_text_model(empty_init=True)

    assert isinstance(text_model, AutoRegressiveTransformer)
    assert text_model.num_layers == 0


def test_merge_audio_features_replaces_audio_placeholders_like_official_model():
    model = Qwen3ASRForConditionalGeneration.from_config(_qwen3_asr_config())
    input_ids = torch.tensor([[10, model.audio_token_id, 11, model.audio_token_id]])
    inputs_embeds = torch.zeros(1, 4, 3)
    audio_features = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    merged = model.merge_audio_features(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        audio_features=audio_features,
    )

    expected = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [4.0, 5.0, 6.0]]])
    torch.testing.assert_close(merged, expected)


def test_merge_audio_features_requires_matching_placeholder_count():
    model = Qwen3ASRForConditionalGeneration.from_config(_qwen3_asr_config())
    input_ids = torch.tensor([[10, model.audio_token_id]])
    inputs_embeds = torch.zeros(1, 2, 3)
    audio_features = torch.randn(2, 3)

    with pytest.raises(ValueError, match="audio feature"):
        model.merge_audio_features(input_ids=input_ids, inputs_embeds=inputs_embeds, audio_features=audio_features)


def test_build_model_returns_thinker_with_osc_text_model_and_audio_tower():
    model = Qwen3ASRForConditionalGeneration.from_config(_qwen3_asr_config(num_text_layers=0, num_audio_layers=1))

    thinker = model.build_model(empty_init=True)

    assert isinstance(thinker.text_model, AutoRegressiveTransformer)
    assert isinstance(thinker.audio_tower, Qwen3ASRAudioEncoder)
    assert thinker.audio_token_id == model.audio_token_id


def test_thinker_prepare_inputs_embeds_merges_precomputed_audio_features():
    model = Qwen3ASRForConditionalGeneration.from_config(_qwen3_asr_config(num_text_layers=0, num_audio_layers=1))
    thinker = model.build_model(empty_init=False)
    input_ids = torch.tensor([[5, model.audio_token_id, 6]])
    audio_features = torch.randn(1, thinker.text_model.embedding.embed.embedding_dim)

    inputs_embeds = thinker.prepare_inputs_embeds(input_ids=input_ids, audio_features=audio_features)

    torch.testing.assert_close(inputs_embeds[0, 1], audio_features[0])
    torch.testing.assert_close(inputs_embeds[0, 0], thinker.text_model.embedding(input_ids.flatten())[0])


class _FakeTextModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = VocabEmbedding(num_embeddings=151680, embedding_dim=4)
        self.seqs = None

    def batch(self, seqs: list[Sequence]):
        self.seqs = seqs
        for seq in seqs:
            seq.append_token(42)
        return seqs


def test_thinker_generate_delegates_prompt_embeds_to_osc_text_model():
    text_model = _FakeTextModel()
    thinker = Qwen3ASRThinker(text_model=text_model, audio_tower=torch.nn.Identity(), audio_token_id=151676)
    input_ids = torch.tensor([[7, 151676, 8]])
    audio_features = torch.ones(1, 4)

    outputs = thinker.generate(input_ids=input_ids, audio_features=audio_features)

    assert outputs == [[42]]
    assert text_model.seqs is not None
    assert text_model.seqs[0].token_ids == [7, 151676, 8, 42]
    assert text_model.seqs[0].prompt_embeds.shape == (3, 4)
    torch.testing.assert_close(text_model.seqs[0].prompt_embeds[1], audio_features[0])


class _FakeThinker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return [[101, 102]]


class _FakeRuntime(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.setup_kwargs = None

    def setup(self, **kwargs):
        self.setup_kwargs = kwargs


def test_qwen3_asr_generate_delegates_to_loaded_thinker():
    model = Qwen3ASRForConditionalGeneration.from_config(_small_qwen3_asr_config())
    model.model = _FakeThinker()
    input_ids = torch.tensor([[1, 2, 3]])
    input_features = torch.randn(1, 128, 10)

    outputs = model.generate(input_ids=input_ids, input_features=input_features)

    assert outputs == [[101, 102]]
    assert model.model.kwargs["input_ids"] is input_ids
    assert model.model.kwargs["input_features"] is input_features


def test_qwen3_asr_setup_uses_configured_runtime_defaults():
    model = Qwen3ASRForConditionalGeneration.from_config(_qwen3_asr_config())
    model.model = Qwen3ASRThinker(
        text_model=_FakeRuntime(),
        audio_tower=torch.nn.Identity(),
        audio_token_id=model.audio_token_id,
    )

    model.setup(gpu_memory_utilization=0.5, device="cuda:0", cuda_graph=False)

    assert model.model.text_model.setup_kwargs["max_model_len"] == 256
    assert model.model.text_model.setup_kwargs["gpu_memory_utilization"] == 0.5
    assert model.model.text_model.setup_kwargs["device"] == "cuda:0"
    assert model.model.text_model.setup_kwargs["cuda_graph"] is False
    assert model.model.text_model.setup_kwargs["dtype"] is torch.bfloat16
    assert model.model.text_model.setup_kwargs["eos"] == [151645, 151643]
    assert model.model.text_model.setup_kwargs["model_name"] == "Qwen3ASRForConditionalGeneration"


def test_qwen3_asr_setup_moves_audio_tower_to_runtime_dtype():
    model = Qwen3ASRForConditionalGeneration.from_config(_qwen3_asr_config())
    audio_tower = torch.nn.Linear(1, 1)
    model.model = Qwen3ASRThinker(
        text_model=_FakeRuntime(),
        audio_tower=audio_tower,
        audio_token_id=model.audio_token_id,
    )

    model.setup(device="cpu", cuda_graph=False)

    assert model.device == torch.device("cpu")
    assert model.dtype is torch.bfloat16
    assert audio_tower.weight.dtype is torch.bfloat16


def test_thinker_generate_accepts_attention_mask_from_processor():
    text_model = _FakeTextModel()
    thinker = Qwen3ASRThinker(text_model=text_model, audio_tower=torch.nn.Identity(), audio_token_id=151676)

    outputs = thinker.generate(
        input_ids=torch.tensor([[7, 151676, 8]]),
        attention_mask=torch.tensor([[1, 1, 1]]),
        audio_features=torch.ones(1, 4),
    )

    assert outputs == [[42]]


def test_thinker_generate_trims_padded_prompt_with_attention_mask():
    text_model = _FakeTextModel()
    thinker = Qwen3ASRThinker(text_model=text_model, audio_tower=torch.nn.Identity(), audio_token_id=151676)

    outputs = thinker.generate(
        input_ids=torch.tensor([[7, 151676, 8, 0]]),
        attention_mask=torch.tensor([[1, 1, 1, 0]]),
        audio_features=torch.ones(1, 4),
    )

    assert outputs == [[42]]
    assert text_model.seqs[0].token_ids == [7, 151676, 8, 42]
    assert text_model.seqs[0].prompt_embeds.shape == (3, 4)


def test_thinker_generate_preserves_batch_input_order_when_runtime_returns_finished_order():
    class ReorderingTextModel(_FakeTextModel):
        def batch(self, seqs: list[Sequence]):
            self.seqs = seqs
            seqs[0].append_token(101)
            seqs[1].append_token(202)
            return list(reversed(seqs))

    text_model = ReorderingTextModel()
    thinker = Qwen3ASRThinker(text_model=text_model, audio_tower=torch.nn.Identity(), audio_token_id=151676)

    outputs = thinker.generate(
        input_ids=torch.tensor([[7, 8], [9, 10]]),
        audio_features=torch.empty(0, 4),
    )

    assert outputs == [[101], [202]]


class _FakeProcessor:
    def __init__(self):
        self.calls = []
        self.templates = []
        self.inputs = None

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        self.templates.append(
            {
                "messages": messages,
                "add_generation_prompt": add_generation_prompt,
                "tokenize": tokenize,
            }
        )
        return "<|im_start|>system\nctx<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n"

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        self.inputs = _MovableInputs(
            {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "input_features": torch.randn(1, 128, 10),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
                "feature_attention_mask": torch.ones(1, 10, dtype=torch.long),
            }
        )
        return self.inputs

    def batch_decode(self, batch_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return ["decoded:" + ",".join(map(str, batch_ids[0]))]


class _MovableInputs(dict):
    def __init__(self, values):
        super().__init__(values)
        self.to_calls = []

    def to(self, target):
        self.to_calls.append(target)
        return self


class _PlainProcessor(_FakeProcessor):
    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "input_features": torch.randn(1, 128, 10),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "feature_attention_mask": torch.ones(1, 10, dtype=torch.long),
        }


class _FakeASRModel:
    def __init__(self):
        self.kwargs = None
        self.device = torch.device("cpu")
        self.dtype = torch.float16

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return [[42, 43]]


def test_asr_wrapper_uses_processor_and_decodes_model_outputs():
    processor = _FakeProcessor()
    model = _FakeASRModel()
    asr = ASR(checkpoint_dir="/tmp/checkpoint", processor=processor, model=model)

    result = asr.transcribe(audio=[0.0, 1.0], sampling_rate=16000, context="ctx", language="English")

    assert result == "decoded:42,43"
    assert processor.templates[0]["messages"] == [
        {"role": "system", "content": "ctx"},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    assert processor.calls[0]["text"] == [
        "<|im_start|>system\nctx<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\nlanguage English<asr_text>"
    ]
    assert processor.calls[0]["audio"][0].tolist() == [0.0, 1.0]
    assert processor.calls[0]["audio"][0].dtype == np.float32
    assert processor.calls[0]["sampling_rate"] == 16000
    assert processor.calls[0]["return_tensors"] == "pt"
    assert processor.calls[0]["padding"] is True
    assert "input_ids" in model.kwargs
    assert "input_features" in model.kwargs
    assert "attention_mask" in model.kwargs
    assert "feature_attention_mask" in model.kwargs


def test_asr_wrapper_moves_processor_inputs_to_model_device_and_dtype():
    processor = _FakeProcessor()
    model = _FakeASRModel()
    asr = ASR(checkpoint_dir="/tmp/checkpoint", processor=processor, model=model)

    asr.transcribe(audio=[0.0, 1.0])

    assert processor.inputs.to_calls == [model.device, model.dtype]


def test_asr_wrapper_moves_plain_dict_tensor_inputs_to_model_device():
    processor = _PlainProcessor()
    model = _FakeASRModel()
    asr = ASR(checkpoint_dir="/tmp/checkpoint", processor=processor, model=model)

    asr.transcribe(audio=[0.0, 1.0])

    assert model.kwargs["input_ids"].device == model.device
    assert model.kwargs["input_features"].device == model.device
    assert model.kwargs["input_features"].dtype == model.dtype
    assert model.kwargs["attention_mask"].dtype == torch.long


def test_asr_wrapper_resamples_audio_to_official_16khz_before_processing():
    processor = _PlainProcessor()
    asr = ASR(checkpoint_dir="/tmp/checkpoint", processor=processor, model=_FakeASRModel())
    audio = np.linspace(-0.5, 0.5, 48000, dtype=np.float32)

    asr.transcribe(audio=audio, sampling_rate=48000)

    processed = processor.calls[0]["audio"][0]
    assert isinstance(processed, np.ndarray)
    assert processed.dtype == np.float32
    assert len(processed) == 16000
    assert processor.calls[0]["sampling_rate"] == 16000


def test_asr_wrapper_parses_unforced_language_tagged_output():
    class TaggedProcessor(_FakeProcessor):
        def batch_decode(self, batch_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
            return ["language Chinese<asr_text>你好"]

    asr = ASR(checkpoint_dir="/tmp/checkpoint", processor=TaggedProcessor(), model=_FakeASRModel())

    result = asr.transcribe(audio=[0.0, 1.0])

    assert result == "你好"
