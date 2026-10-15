"""Qwen3-VL-4B text encoder loaded from a single ComfyUI file.

This is the file loader and 12-layer tap from comfyui-iris (Load Iris CLIP),
wired so Iris-3B's sampler still receives a fixed 300-token stack of hidden
states. It does not download a Hugging Face repo. Vision weights in a full
Qwen3-VL file are ignored; only the language stack is used.
"""

import os
from pathlib import Path

import torch

import comfy.model_management
import comfy.sd
import comfy.supported_models_base
import comfy.utils
from comfy import sd1_clip
from comfy.text_encoders.llama import BaseLlama, Llama2_, Qwen3VL_4BConfig
from comfy.text_encoders.qwen3vl import Qwen3VLTokenizer

from bucky_iris3b.text.base import TextEncoding

HIDDEN_LAYERS = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]
PREFIX = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
TEXT_LENGTH = 300
_LAYER_WEIGHTS = (
    "input_layernorm",
    "post_attention_layernorm",
    "self_attn.q_norm",
    "self_attn.k_norm",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


class IrisTokenizer(Qwen3VLTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, model_type="qwen3vl_4b")
        tokenizer = self.qwen3vl_4b.tokenizer
        self.prefix_ids = tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix_ids = tokenizer.encode(SUFFIX, add_special_tokens=False)

    def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
        ids = self.qwen3vl_4b.tokenizer.encode(text, add_special_tokens=False)
        budget = TEXT_LENGTH - len(self.suffix_ids)
        ids = self.prefix_ids + ids[:budget] + self.suffix_ids
        ids += [151643] * (len(self.prefix_ids) + TEXT_LENGTH - len(ids))
        pairs = [(token, 1.0, i) if return_word_ids else (token, 1.0) for i, token in enumerate(ids)]
        return {"qwen3vl_4b": [pairs]}


class IrisLanguageModel(BaseLlama, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = Qwen3VL_4BConfig(**config_dict)
        self.num_layers = config.num_hidden_layers
        self.model = Llama2_(config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype


class IrisQwenClip(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(
            device=device,
            dtype=dtype,
            model_options=model_options,
            layer="hidden",
            layer_idx=-1,
            textmodel_json_config={},
            special_tokens={"pad": 151643},
            layer_norm_hidden_state=False,
            model_class=IrisLanguageModel,
            enable_attention_masks=True,
            return_attention_masks=True,
        )

    def encode_token_weights(self, token_weight_pairs):
        self.set_clip_options({"layer": HIDDEN_LAYERS})
        # Tap at block i's input. That matches Hugging Face hidden_states[i].
        states, _, extra = self.encode([[token[0] for token in row] for row in token_weight_pairs])
        states = states.transpose(1, 2)
        start = states.shape[1] - TEXT_LENGTH
        states = states[:, start:]
        mask = extra["attention_mask"][:, start:]
        states = states * mask[:, :, None, None]
        device = comfy.model_management.intermediate_device()
        return states.to(device), None, {"attention_mask": mask.to(device)}


class IrisTE(sd1_clip.SD1ClipModel):
    """Language stack only. encode_token_weights stays on IrisQwenClip."""

    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype, name="qwen3vl_4b", clip_model=IrisQwenClip, model_options=model_options)

    def load_sd(self, sd):
        missing, unexpected = super().load_sd(sd)
        missing = [key for key in missing if key.startswith("model.")]
        if missing or unexpected:
            raise ValueError(f"Invalid Qwen3-VL-4B text weights: missing={missing}, unexpected={unexpected}")
        return missing, unexpected


def _reject_unless_qwen3vl_4b(sd: dict, name: str) -> None:
    """Full Qwen3-VL files carry a vision tower. Qwen3-4B (Z-Image) does not, and Iris was not trained on it."""
    has_visual = any(".visual." in key or key.startswith("visual.") for key in sd)
    if has_visual:
        for key, value in sd.items():
            if key.endswith("merger.linear_fc2.weight") and value.shape[0] != 2560:
                raise ValueError(f"{name} is a Qwen3-VL text encoder, but not the 4B one Iris was trained with.")
        return
    low = os.path.basename(name).lower()
    if "qwen3vl" in low or "qwen3-vl" in low or "qwen3_vl" in low:
        return
    raise ValueError(
        f"{name} is not a Qwen3-VL-4B text encoder. Iris reads 12 hidden layers from Qwen3-VL-4B. "
        "Pick a file such as qwen3vl_4b_fp8_scaled.safetensors or qwen3_vl_4b_nvfp4_full.safetensors."
    )


def _language_state(sd: dict) -> dict:
    prefixes = ("model.language_model.", "language_model.", "model.")
    prefix = next((item for item in prefixes if item + "norm.weight" in sd), None)
    if prefix is None:
        raise ValueError("Iris requires compatible Qwen3-VL-4B language weights.")
    text_sd = comfy.utils.state_dict_prefix_replace(sd, {prefix: "model."}, filter_keys=True)
    text_sd = {key: value for key, value in text_sd.items() if not key.startswith(("model.visual.", "model.lm_head."))}
    signature = {
        "model.norm.weight": 2560,
        "model.layers.0.self_attn.q_norm.weight": 128,
        "model.layers.0.self_attn.k_proj.weight": 1024,
        "model.layers.35.self_attn.q_proj.weight": 4096,
    }
    if any(key not in text_sd or text_sd[key].shape[0] != size for key, size in signature.items()) or any(
        key.startswith("model.layers.36.") for key in text_sd
    ):
        raise ValueError("Iris requires Qwen3-VL-4B's 36-layer, 2560-wide language architecture.")
    required = ["model.embed_tokens.weight", "model.norm.weight"]
    required += [f"model.layers.{layer}.{name}.weight" for layer in range(36) for name in _LAYER_WEIGHTS]
    missing = [key for key in required if key not in text_sd]
    if missing:
        raise ValueError(f"Incomplete Qwen3-VL-4B language weights: missing={missing}")
    return text_sd


class FileTextEncoder:
    """Same encode / null / to surface the old folder encoder had, backed by a ComfyUI CLIP."""

    def __init__(self, clip, nbytes: int, cache_path: Path):
        self.clip = clip
        self.nbytes = nbytes
        self._cache_path = cache_path

    def encode(self, prompts: list[str]) -> TextEncoding:
        if len(prompts) != 1:
            raise ValueError("Iris text encode expects one prompt at a time")
        tokens = self.clip.tokenize(prompts[0])
        out = self.clip.encode_from_tokens(tokens, return_dict=True)
        return TextEncoding(out["cond"].detach().cpu(), out["attention_mask"].detach().cpu())

    def null(self, negative_prompt: str = "") -> TextEncoding:
        cache = self._cache_path if negative_prompt == "" else None
        if cache is not None and cache.is_file():
            payload = torch.load(cache, map_location="cpu", weights_only=True)
            return TextEncoding(payload["embeddings"], payload["mask"])
        encoding = self.encode([negative_prompt])
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"embeddings": encoding.embeddings, "mask": encoding.mask}, cache)
        return encoding

    def to(self, device) -> "FileTextEncoder":
        # The CLIP patcher loads itself inside encode(). Moving back to CPU drops only this encoder.
        if torch.device(device).type == "cpu":
            loaded = comfy.model_management.current_loaded_models
            keep = [item for item in loaded if item.model is not self.clip.patcher]
            comfy.model_management.free_memory(1e30, comfy.model_management.get_torch_device(), keep)
        return self


def load_file_encoder(path: str, cache_dir: str) -> FileTextEncoder:
    name = os.path.basename(path)
    sd, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
    _reject_unless_qwen3vl_4b(sd, name)
    sd, _ = comfy.utils.convert_old_quants(sd, model_prefix="", metadata=metadata)
    text_sd = _language_state(sd)
    quant = comfy.utils.detect_layer_quantization(text_sd, "")
    model_options = {"quantization_metadata": quant} if quant is not None else {}
    nbytes = sum(value.numel() * value.element_size() for value in text_sd.values() if torch.is_tensor(value))
    target = comfy.supported_models_base.ClipTarget(IrisTokenizer, IrisTE)
    clip = comfy.sd.CLIP(
        target,
        embedding_directory=None,
        parameters=comfy.utils.calculate_parameters(text_sd),
        state_dict=[text_sd],
        model_options=model_options,
    )
    stat = os.stat(path)
    cache_path = Path(cache_dir) / f"null_embed_{Path(name).stem}_{stat.st_size}_{int(stat.st_mtime)}_{TEXT_LENGTH}.pth"
    encoder = FileTextEncoder(clip, nbytes, cache_path)
    encoder.to("cpu")
    return encoder
