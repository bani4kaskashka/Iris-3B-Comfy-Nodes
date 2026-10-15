"""ComfyUI nodes for Iris-3B (speridlabs/iris-3b), a pixel-space text-to-image DiT.

Iris has no VAE: the sampler denoises RGB pixels directly and returns an IMAGE.
Its text encoder is the language stack of Qwen3-VL-4B (12 hidden layers are
stacked). Point the loader at a single file in models/text_encoders, the same
way other workflows pick a CLIP. The 4x upscaler is Iris fine-tuned for
restoration: one forward pass per 1024 px tile, conditioned on a stored empty
prompt, so it needs no text encoder.

Every model lives in system RAM between runs and only visits the GPU for its
own step, so they never share VRAM with each other. Any ComfyUI-managed models
are asked to make room first.

Default files, downloaded from the Hub on first use when missing:

    models/diffusion_models/iris-3b.safetensors            text to image
    models/diffusion_models/iris-3b-upscaler.safetensors   4x upscaler
    models/diffusion_models/iris-3b-depth.safetensors      monocular depth
    models/text_encoders/<Qwen3-VL-4B file>                one text encoder file
    models/loras/iris-3b-turbo-r2-1000.safetensors         4-step Turbo LoRA (optional)

The configs come with this pack; the upscaler's and depth model's empty-prompt
embeddings (36 MB each) are fetched next to them on first use. A
Hub-layout folder (model.safetensors next to config.yaml) also works.
"""

import copy
import gc
import json
import logging
import os
import shutil
from contextlib import contextmanager

import torch
from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import load_file

import comfy.model_management as mm
import comfy.utils
import folder_paths

from bucky_iris3b.config import inference_config
from bucky_iris3b.flow.schedule import shift_sigma
from bucky_iris3b.flow.solver import FlowDPMSolver
from bucky_iris3b.models.dit import IrisDiT
from bucky_iris3b.text.base import TextEncoding

from .text_encoder import load_file_encoder

log = logging.getLogger("ComfyUI-Iris3B")

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = os.path.join(HERE, "configs")
NULL_CACHE_DIR = os.path.join(HERE, "cache")
GB = 1024**3

IRIS_REPO = "speridlabs/iris-3b"
HF = "https://huggingface.co"

# default file name -> (Hub file, bundled config)
T2I_FILE = "iris-3b.safetensors"
UPSCALER_FILE = "iris-3b-upscaler.safetensors"
DEPTH_FILE = "iris-3b-depth.safetensors"
STOCK = {
    T2I_FILE: ("model.safetensors", "iris-3b.yaml"),
    UPSCALER_FILE: ("upscaler/model.safetensors", "iris-3b-upscaler.yaml"),
    DEPTH_FILE: ("depth/model.safetensors", "iris-3b-depth.yaml"),
}


def hub_url(path: str, repo: str = IRIS_REPO) -> str:
    return f"{HF}/{repo}/resolve/main/{path}"


def _empty_cache() -> None:
    gc.collect()
    mm.soft_empty_cache()


def _iris_files(kind: str) -> list[str]:
    """Iris weights in diffusion_models (names with "iris" in them) of one kind:
    "t2i", "upscaler" or "depth"."""
    out = []
    for name in folder_paths.get_filename_list("diffusion_models"):
        low = name.lower().replace("\\", "/")
        if "iris" not in low or "empty_prompt" in low:
            continue
        found = "upscaler" if "upscaler" in low else "depth" if "depth" in low else "t2i"
        if found == kind:
            out.append(name)
    return out


def _with_default(found: list[str], default: str) -> list[str]:
    # the default is listed only once it exists, so the frontend's missing-model
    # check can offer the download; VALIDATE_INPUTS still lets a missing default
    # through and the loader fetches it
    return [default] + [f for f in found if f != default] if default in found else found


def _fetch(repo: str, path: str, target: str) -> str:
    """Download one Hub file to ``target`` (any name)."""
    from huggingface_hub import hf_hub_download

    os.makedirs(os.path.dirname(target), exist_ok=True)
    staging = os.path.join(os.path.dirname(target), ".iris-download")
    log.info("downloading %s/%s to %s", repo, path, target)
    got = hf_hub_download(repo, path, local_dir=staging)
    shutil.move(got, target)
    shutil.rmtree(staging, ignore_errors=True)
    return target


def _weights_path(name: str) -> str:
    path = folder_paths.get_full_path("diffusion_models", name)
    if path:
        return path
    if name not in STOCK:
        raise FileNotFoundError(f"{name} not found in models/diffusion_models")
    target = os.path.join(folder_paths.get_folder_paths("diffusion_models")[0], name)
    return _fetch(IRIS_REPO, STOCK[name][0], target)


def _config_path(weights: str, fallback: str) -> str:
    """config.yaml next to Hub-layout weights, else the bundled one."""
    side = os.path.join(os.path.dirname(weights), "config.yaml")
    if os.path.basename(weights) == "model.safetensors" and os.path.isfile(side):
        return side
    return os.path.join(CONFIGS, fallback)


# Per-row int8 used by Iris-3b-int8.safetensors. Each Linear is stored as
# qweight int8 [out, in] and scale fp32 [out, 1]; the DiT is plain nn.Linear.
ROWWISE_INT8 = "iris-rowwise-int8-v1"


def _row_scale(qweight: torch.Tensor, scale: torch.Tensor) -> bool:
    if qweight.ndim != 2 or qweight.dtype != torch.int8:
        return False
    if scale.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return False
    rows = qweight.shape[0]
    return scale.shape == (rows, 1) or scale.shape == (rows,)


def _load_weights(path: str, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    # Cast tensor by tensor so an fp32 file never sits in RAM whole. A row-wise
    # int8 file is expanded here (qweight * scale -> weight) and the packed
    # tensors are dropped before the next layer is read.
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        packed = {key for key in keys if key.endswith(".qweight")}
        if not packed:
            return {key: f.get_tensor(key).to(dtype) for key in keys}

        fmt = (f.metadata() or {}).get("format") or ""
        if fmt and fmt != ROWWISE_INT8:
            raise ValueError(
                f"{os.path.basename(path)} is quantized as {fmt!r}. "
                f"This loader only expands {ROWWISE_INT8} (int8 qweight times a per-row scale)."
            )

        present = set(keys)
        weights: dict[str, torch.Tensor] = {}
        for key in keys:
            if key.endswith(".qweight"):
                stem = key[: -len("qweight")]
                scale_key = stem + "scale"
                plain = stem + "weight"
                if scale_key not in present:
                    raise ValueError(f"{os.path.basename(path)} has {key} without {scale_key}")
                if plain in present:
                    raise ValueError(f"{os.path.basename(path)} has both {plain} and {key}")
                qweight = f.get_tensor(key)
                scale = f.get_tensor(scale_key)
                if not _row_scale(qweight, scale):
                    raise ValueError(
                        f"{os.path.basename(path)} {key} is {tuple(qweight.shape)} {qweight.dtype}, "
                        f"scale {tuple(scale.shape)} {scale.dtype}; expected int8 [out, in] and fp scale [out, 1]"
                    )
                weights[plain] = (qweight.float() * scale.float()).to(dtype)
                del qweight, scale
            elif key.endswith(".scale") and (key[: -len("scale")] + "qweight") in packed:
                continue
            else:
                weights[key] = f.get_tensor(key).to(dtype)
        name = os.path.basename(path)
        label = fmt or "row-wise int8"
        log.info("expanded %s (%s) to %s Linear weights", name, label, dtype)
        return weights


def _build_dit(weights_path: str, config_path: str, dtype: torch.dtype) -> tuple[IrisDiT, object, dict | None]:
    raw = OmegaConf.to_container(OmegaConf.load(config_path))
    task = raw.pop("task", None)
    cfg = inference_config(raw)
    weights = _load_weights(weights_path, dtype)
    with torch.device("meta"):
        model = IrisDiT(cfg.model)
    model.load_state_dict(weights, strict=True, assign=True)
    del weights
    return model.train(False).requires_grad_(False).to(dtype=dtype), cfg, task


def _nbytes(module: torch.nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in module.parameters())


DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
DTYPE_INPUT = (list(DTYPES), {
    "default": "bf16",
    "tooltip": "bf16 = 6 GB on the GPU. fp32 = 12 GB (the shipped precision) and needs well over 16 GB VRAM",
})


class IrisModel:
    def __init__(self, model: IrisDiT, cfg, dtype: torch.dtype, name: str):
        self.model, self.cfg, self.dtype, self.name = model, cfg, dtype, name
        self.nbytes = _nbytes(model)
        # [{"layers": {prefix: (down, up)}, "scale": float, "meta": dict}, ...] from the LoRA loaders
        self.loras = []


def _is_turbo(model: "IrisModel") -> bool:
    return any(lora["meta"].get("turbo_steps") for lora in model.loras)


def _read_lora(model: "IrisModel", path: str, name: str, strength: float) -> dict:
    """An Iris LoRA file ("<layer>.lora_down" / "<layer>.lora_up" with rank/alpha metadata)."""
    with safe_open(path, framework="pt") as f:
        raw = f.metadata() or {}
    if "rank" not in raw or "alpha" not in raw:
        raise ValueError(f"{name} is not an Iris LoRA (no rank/alpha metadata)")
    meta = {"rank": int(raw["rank"]), "alpha": float(raw["alpha"]), **json.loads(raw.get("extra") or "{}")}
    tensors = load_file(path)
    layers = {}
    for key in tensors:
        if not key.endswith(".lora_down"):
            continue
        prefix = key[: -len(".lora_down")]
        try:
            linear = model.model.get_submodule(prefix)
        except AttributeError as exc:
            raise ValueError(f"{name} targets {prefix}, which this Iris model lacks") from exc
        if not isinstance(linear, torch.nn.Linear):
            raise ValueError(f"{name}: {prefix} is not a Linear layer")
        up = tensors[prefix + ".lora_up"]
        if up.abs().max() == 0:
            continue  # e.g. the all-zero adaLN-core pairs in the Turbo releases
        layers[prefix] = (tensors[key], up)
    return {"layers": layers, "scale": strength * meta["alpha"] / meta["rank"], "meta": meta, "name": name}


def _with_lora(model: "IrisModel", lora: dict) -> "IrisModel":
    out = copy.copy(model)  # shares the cached DiT; LoRAs only apply inside the samplers
    out.loras = [*model.loras, lora]
    return out


@contextmanager
def _live_lora(model: "IrisModel", device):
    """Adds the attached LoRAs to their Linear layers for the length of a run with forward
    hooks, so the cached base weights are never modified and no second copy is made."""
    handles = []
    try:
        for lora in model.loras:
            scale = lora["scale"]
            if scale == 0:
                continue
            for prefix, (down, up) in lora["layers"].items():
                linear = model.model.get_submodule(prefix)
                d = down.to(device, model.dtype)
                u = (up * scale).to(device, model.dtype)

                def hook(_module, args, out, d=d, u=u):
                    return out + ((args[0].to(d.dtype) @ d.t()) @ u.t()).to(out.dtype)

                handles.append(linear.register_forward_hook(hook))
        yield
    finally:
        for h in handles:
            h.remove()


class IrisTextEncoder:
    def __init__(self, encoder):
        self.encoder = encoder
        self.nbytes = encoder.nbytes


class IrisModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (_with_default(_iris_files("t2i"), T2I_FILE), {
                    "tooltip": f"Iris-3B weights in models/diffusion_models. {T2I_FILE} is downloaded "
                               "on first use if missing (12 GB)",
                }),
                "weight_dtype": DTYPE_INPUT,
            }
        }

    RETURN_TYPES = ("IRIS_MODEL",)
    FUNCTION = "load"
    CATEGORY = "Iris-3B"

    @classmethod
    def VALIDATE_INPUTS(cls, model_name):
        return True  # a missing default is downloaded by load()

    def load(self, model_name, weight_dtype):
        weights = _weights_path(model_name)
        model, cfg, task = _build_dit(weights, _config_path(weights, STOCK[T2I_FILE][1]), DTYPES[weight_dtype])
        if task is not None:
            raise ValueError(f"{model_name} is an Iris {task['name']} model, not the text-to-image one")
        log.info("loaded %s as %s", model_name, weight_dtype)
        return (IrisModel(model, cfg, DTYPES[weight_dtype], model_name),)


def _text_encoder_files() -> list[str]:
    """Qwen3-VL-4B files first, then every other text encoder, same list as a CLIP loader."""
    names = folder_paths.get_filename_list("text_encoders")

    def rank(name: str) -> tuple:
        low = name.lower()
        if "qwen3vl" in low or "qwen3-vl" in low or "qwen3_vl" in low:
            return (0 if "4b" in low else 1, low)
        return (2, low)

    return sorted(names, key=rank)


class IrisTextEncoderLoader:
    @classmethod
    def INPUT_TYPES(cls):
        names = _text_encoder_files() or ["(put a Qwen3-VL-4B file in models/text_encoders)"]
        return {
            "required": {
                "text_encoder": (names, {
                    "tooltip": "A Qwen3-VL-4B file in models/text_encoders, such as "
                               "qwen3vl_4b_fp8_scaled.safetensors or qwen3_vl_4b_nvfp4_full.safetensors. "
                               "fp8, nvfp4 and int8 files are fine. Nothing is downloaded.",
                }),
            }
        }

    RETURN_TYPES = ("IRIS_TEXT_ENCODER",)
    FUNCTION = "load"
    CATEGORY = "Iris-3B"

    def load(self, text_encoder):
        path = folder_paths.get_full_path("text_encoders", text_encoder)
        if not path:
            raise FileNotFoundError(
                f"{text_encoder} is not a text encoder file in models/text_encoders. "
                "Pick a Qwen3-VL-4B file. Iris does not download a Hugging Face folder."
            )
        encoder = load_file_encoder(path, NULL_CACHE_DIR)
        log.info("loaded text encoder %s", text_encoder)
        return (IrisTextEncoder(encoder),)


class IrisTextEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_encoder": ("IRIS_TEXT_ENCODER",),
                "positive": ("STRING", {"multiline": True, "default": "a red fox sleeping in fresh snow, golden hour"}),
                "negative": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "empty = the null prompt Iris was trained with (recommended)",
                }),
            }
        }

    RETURN_TYPES = ("IRIS_CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "Iris-3B"

    @torch.no_grad()
    def encode(self, text_encoder, positive, negative):
        enc = text_encoder.encoder
        device = mm.get_torch_device()
        mm.free_memory(text_encoder.nbytes + 1 * GB, device)
        enc.to(device)
        try:
            cond = enc.encode([positive])
            null = enc.null(negative)
            result = {
                "cond": TextEncoding(cond.embeddings.cpu(), cond.mask.cpu()),
                "null": TextEncoding(null.embeddings.cpu(), null.mask.cpu()),
            }
        finally:
            enc.to("cpu")
            _empty_cache()
        return (result,)


MATCH_IMAGE = "match input image (about 1 MP)"
# name -> (width, height); all about one megapixel, the resolution Iris was fine-tuned at
RESOLUTIONS = {
    "1:1 square (1024x1024)": (1024, 1024),
    "5:4 (1120x896)": (1120, 896),
    "4:5 (896x1120)": (896, 1120),
    "4:3 (1152x864)": (1152, 864),
    "3:4 (864x1152)": (864, 1152),
    "3:2 (1216x832)": (1216, 832),
    "2:3 (832x1216)": (832, 1216),
    "16:9 (1344x768)": (1344, 768),
    "9:16 (768x1344)": (768, 1344),
    "21:9 (1536x640)": (1536, 640),
    "9:21 (640x1536)": (640, 1536),
}
QUALITY = {"fast (30 steps)": 30, "balanced (50 steps)": 50, "best (100 steps)": 100}
GUIDANCE = {"loose (cfg 2)": 2.0, "normal (cfg 3)": 3.0, "strict (cfg 4.5)": 4.5}
MODES = ["text to image", "image to image"]
STRENGTH = {"subtle (denoise 0.3)": 0.3, "medium (denoise 0.5)": 0.5, "strong (denoise 0.7)": 0.7}
MODE_INPUT = (MODES, {"default": "text to image",
                      "tooltip": "image to image starts from the connected image instead of pure noise"})
IMAGE_INPUT = ("IMAGE", {"tooltip": "source for image to image (ignored in text to image)"})


def _round16(v: float) -> int:
    return max(256, int(round(v / 16)) * 16)


def prepare_image(image: torch.Tensor, width: int | None, height: int | None) -> torch.Tensor:
    """IMAGE [B, H, W, C] in 0..1 -> [B, 3, H, W] in -1..1. With a target size the image
    is scaled to cover it and center-cropped; without one it keeps its aspect ratio at
    about one megapixel."""
    F = torch.nn.functional
    x = image[..., :3].permute(0, 3, 1, 2).float()
    h, w = x.shape[-2:]
    if width is None or height is None:
        scale = (1024 * 1024 / (h * w)) ** 0.5
        width, height = _round16(w * scale), _round16(h * scale)
    scale = max(width / w, height / h)
    size = (max(height, round(h * scale)), max(width, round(w * scale)))
    x = F.interpolate(x, size=size, mode="bicubic", align_corners=False, antialias=True)
    top, left = (size[0] - height) // 2, (size[1] - width) // 2
    return (x[..., top:top + height, left:left + width].clamp(0, 1) * 2 - 1).contiguous()

SEED = ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True})
BATCH = ("INT", {"default": 1, "min": 1, "max": 8, "tooltip": "images per run"})


@torch.no_grad()
def run_sampler(model, conditioning, width, height, steps, cfg, seed, batch_size,
                shift=4.0, solver_order=2, cfg_start=0.0, cfg_end=1.0, keep_model_on_gpu=False,
                init_image=None, denoise=1.0):
    """``init_image`` ([B, 3, H, W] in -1..1) turns this into image to image: the run
    takes the last ``steps`` steps of a ``steps / denoise`` grid, starting from the
    image noised to that grid's first level. A batch of one image is repeated."""
    if init_image is not None:
        height, width = init_image.shape[-2:]
        if init_image.shape[0] > 1:
            batch_size = init_image.shape[0]
    if _is_turbo(model):
        raise ValueError("this model has a Turbo LoRA attached: connect it to Iris Sampler (Turbo) instead")
    p = model.cfg.model.patch_size
    if width % p or height % p:
        raise ValueError(f"width and height must be multiples of {p}")
    device = mm.get_torch_device()
    dit = model.model
    activations = int(2.5 * GB * batch_size * (width * height) / (1024 * 1024))
    mm.free_memory(model.nbytes + activations, device)
    dit.to(device)

    dtype = model.dtype
    cond = conditioning["cond"].embeddings.to(device, dtype).expand(batch_size, *([-1] * (conditioning["cond"].embeddings.ndim - 1)))
    cond_mask = conditioning["cond"].mask.to(device).expand(batch_size, -1)
    uncond = conditioning["null"].embeddings.to(device, dtype).expand(batch_size, *([-1] * (conditioning["null"].embeddings.ndim - 1)))
    null_mask = conditioning["null"].mask.to(device).expand(batch_size, -1)
    cfg_mask = torch.cat([null_mask, cond_mask], dim=0)

    # same noise as scripts/sample.py for a given seed (drawn on the GPU)
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(batch_size, model.cfg.model.in_channels, height, width, generator=generator, device=device)
    img2img = init_image is not None and denoise < 1.0
    total = max(steps, round(steps / max(denoise, 1e-3))) if img2img else steps

    def model_fn(xin, t_model, y):
        mm.throw_exception_if_processing_interrupted()
        out = dit(xin.to(dtype), t_model, y, y_mask=cond_mask if y is cond else cfg_mask)
        return out.x.float()

    solver = FlowDPMSolver(
        model_fn,
        num_timesteps=model.cfg.flow.num_train_timesteps,
        cfg_scale=cfg,
        cfg_interval=(cfg_start, cfg_end),
        prediction=model.cfg.flow.prediction,
    )
    previews = _previews_enabled()
    pbar = comfy.utils.ProgressBar(steps)
    autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    try:
        with autocast, _live_lora(model, device):
            # FlowDPMSolver.sample() with a per-step callback; x0 is already an image
            grid = solver.time_grid(total, shift)[-(steps + 1):]
            if img2img:
                t0 = grid[0]
                image = init_image.to(device, torch.float32).expand(batch_size, -1, -1, -1)
                x = (1.0 - t0) * image + t0 * x
            history = []
            for i in range(1, steps + 1):
                s, t = grid[i - 1], grid[i]
                x0 = solver._pred_x0(x, s, cond, uncond)
                history.append((s, x0))
                step_order = min(i, solver_order, steps + 1 - i)
                if step_order == 1:
                    x = solver._first_order(x, s, t, x0)
                else:
                    x = solver._second_order(x, history[-2], history[-1], t)
                if len(history) > 2:
                    history.pop(0)
                pbar.update_absolute(i, steps, _preview(x0) if previews and i < steps else None)
        images = ((x.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).float().cpu()
    finally:
        if not keep_model_on_gpu:
            dit.to("cpu")
        del x, cond, uncond
        _empty_cache()
    return images


class IrisSamplerSimple:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("IRIS_MODEL",),
                "conditioning": ("IRIS_CONDITIONING",),
                "mode": MODE_INPUT,
                "resolution": ([MATCH_IMAGE] + list(RESOLUTIONS), {
                    "default": "1:1 square (1024x1024)",
                    "tooltip": "in image to image the input is cropped to this size, or keeps its own "
                               "aspect ratio with match input image",
                }),
                "quality": (list(QUALITY), {"default": "fast (30 steps)",
                                            "tooltip": "100 steps is the model card default, about 3x slower than fast"}),
                "prompt_adherence": (list(GUIDANCE), {"default": "normal (cfg 3)",
                                                      "tooltip": "how strictly the image follows the prompt"}),
                "image_strength": (list(STRENGTH), {
                    "default": "medium (denoise 0.5)",
                    "tooltip": "image to image only: how much the image may change",
                }),
                "seed": SEED,
                "batch_size": BATCH,
            },
            "optional": {"image": IMAGE_INPUT},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sample"
    CATEGORY = "Iris-3B"
    DESCRIPTION = "Iris-3B sampler with presets. Use the Advanced sampler for exact sizes and solver settings."

    def sample(self, model, conditioning, mode, resolution, quality, prompt_adherence, image_strength, seed,
               batch_size, image=None):
        init = None
        if mode == "image to image":
            if image is None:
                raise ValueError("image to image needs an image connected to the sampler")
            size = (None, None) if resolution == MATCH_IMAGE else RESOLUTIONS[resolution]
            init = prepare_image(image, *size)
            width, height = init.shape[-1], init.shape[-2]
        else:
            width, height = RESOLUTIONS.get(resolution, (1024, 1024))
        images = run_sampler(model, conditioning, width, height, QUALITY[quality], GUIDANCE[prompt_adherence],
                             seed, batch_size, init_image=init, denoise=STRENGTH[image_strength])
        return (images,)


class IrisSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("IRIS_MODEL",),
                "conditioning": ("IRIS_CONDITIONING",),
                "steps": ("INT", {"default": 50, "min": 1, "max": 500, "tooltip": "the model card default is 100"}),
                "cfg": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "seed": SEED,
                "batch_size": BATCH,
                "mode": MODE_INPUT,
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01,
                                      "tooltip": "image to image only: 1 ignores the image content, lower "
                                                 "values keep more of it"}),
                "shift": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 20.0, "step": 0.1,
                                    "tooltip": "flow shift (trained with 4.0)"}),
                "solver_order": ("INT", {"default": 2, "min": 1, "max": 2, "tooltip": "DPM-Solver++ order"}),
                "cfg_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "CFG applies only for noise levels t in (cfg_start, cfg_end); 1 = pure noise"}),
                "cfg_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16,
                                  "tooltip": "in image to image the input is scaled and cropped to this size"}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}),
                "keep_model_on_gpu": ("BOOLEAN", {"default": False,
                                                  "tooltip": "skip moving the 6 GB DiT back to RAM after sampling"}),
            },
            "optional": {"image": IMAGE_INPUT},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sample"
    CATEGORY = "Iris-3B"

    def sample(self, model, conditioning, steps, cfg, seed, batch_size, mode, denoise, shift, solver_order,
               cfg_start, cfg_end, width, height, keep_model_on_gpu, image=None):
        init = None
        if mode == "image to image":
            if image is None:
                raise ValueError("image to image needs an image connected to the sampler")
            init = prepare_image(image, width, height)
        images = run_sampler(model, conditioning, width, height, steps, cfg, seed, batch_size,
                             shift, solver_order, cfg_start, cfg_end, keep_model_on_gpu,
                             init_image=init, denoise=denoise)
        return (images,)


TURBO_REPO = "pentacoxian-dev/iris-3b-turbo-lora"
TURBO_FILES = {
    "iris-3b-turbo-r2-1000.safetensors": "r2-1000/student.safetensors",
    "iris-3b-turbo-r1-1500.safetensors": "r1-1500/student.safetensors",
}
TURBO_DEFAULT = "iris-3b-turbo-r2-1000.safetensors"


def _turbo_files() -> list[str]:
    found = [f for f in folder_paths.get_filename_list("loras") if "iris" in f.lower() and "turbo" in f.lower()]
    return [TURBO_DEFAULT] + [f for f in found if f != TURBO_DEFAULT] if TURBO_DEFAULT in found else found


class IrisLoRA:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("IRIS_MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"), {
                    "tooltip": "an Iris LoRA in models/loras (made with iris-3b LoRA trainers)"}),
                "strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 3.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IRIS_MODEL",)
    FUNCTION = "load"
    CATEGORY = "Iris-3B"
    DESCRIPTION = "Adds an Iris LoRA to the model. Chain several of these to stack LoRAs."

    def load(self, model, lora_name, strength):
        lora = _read_lora(model, folder_paths.get_full_path_or_raise("loras", lora_name), lora_name, strength)
        log.info("LoRA %s: %d layers at strength %g", lora_name, len(lora["layers"]), strength)
        return (_with_lora(model, lora),)


class IrisTurboLoRA:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("IRIS_MODEL",),
                "lora_name": (_turbo_files(), {
                    "tooltip": f"Iris Turbo LoRA in models/loras. {TURBO_DEFAULT} (recommended) and "
                               "iris-3b-turbo-r1-1500.safetensors are downloaded on first use if missing (460 MB)",
                }),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IRIS_MODEL",)
    FUNCTION = "load"
    CATEGORY = "Iris-3B"
    DESCRIPTION = ("4-step Turbo LoRA for Iris-3B (DMD2 distillation by pentaCoxian). Guidance is baked in, "
                   "so use it with Iris Sampler (Turbo).")

    @classmethod
    def VALIDATE_INPUTS(cls, lora_name):
        return True  # a missing stock file is downloaded by load()

    def load(self, model, lora_name, strength):
        path = folder_paths.get_full_path("loras", lora_name)
        if not path:
            if lora_name not in TURBO_FILES:
                raise FileNotFoundError(f"{lora_name} not found in models/loras")
            target = os.path.join(folder_paths.get_folder_paths("loras")[0], lora_name)
            path = _fetch(TURBO_REPO, TURBO_FILES[lora_name], target)
        lora = _read_lora(model, path, lora_name, strength)
        if not lora["meta"].get("turbo_steps"):
            raise ValueError(f"{lora_name} is not a Turbo LoRA; use Load Iris LoRA for regular ones")
        log.info("Turbo LoRA %s: %d layers, %s steps", lora_name, len(lora["layers"]), lora["meta"]["turbo_steps"])
        return (_with_lora(model, lora),)


class IrisSamplerTurbo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("IRIS_MODEL", {"tooltip": "an Iris model with Load Iris Turbo LoRA applied"}),
                "conditioning": ("IRIS_CONDITIONING", {"tooltip": "only the positive prompt is used: guidance is baked in"}),
                "resolution": (list(RESOLUTIONS), {"default": "1:1 square (1024x1024)",
                                                   "tooltip": "the Turbo LoRA was trained at about one megapixel"}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 16,
                                  "tooltip": "the LoRA was distilled for 4 (one model pass each)"}),
                "seed": SEED,
                "batch_size": BATCH,
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sample"
    CATEGORY = "Iris-3B"
    DESCRIPTION = "Few-step sampler for the Iris Turbo LoRA: one model pass per step, no CFG."

    @torch.no_grad()
    def sample(self, model, conditioning, resolution, steps, seed, batch_size):
        turbo = next((lora for lora in model.loras if lora["meta"].get("turbo_steps")), None)
        if turbo is None:
            raise ValueError("Iris Sampler (Turbo) needs a model from Load Iris Turbo LoRA")
        width, height = RESOLUTIONS[resolution]
        shift = float(turbo["meta"].get("shift", 4.0))
        device = mm.get_torch_device()
        dit, dtype = model.model, model.dtype
        mm.free_memory(model.nbytes + int(1.5 * GB * batch_size), device)
        dit.to(device)
        cond = conditioning["cond"].embeddings.to(device, dtype)
        cond = cond.expand(batch_size, *([-1] * (cond.ndim - 1)))
        mask = conditioning["cond"].mask.to(device).expand(batch_size, -1)
        # upstream turbo schedule: evenly spaced flow times from pure noise, shifted
        times = shift_sigma(1.0 - torch.arange(steps, dtype=torch.float64) / steps, shift).tolist()
        generator = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn(batch_size, model.cfg.model.in_channels, height, width, generator=generator, device=device)
        previews, pbar = _previews_enabled(), comfy.utils.ProgressBar(steps)
        autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
        try:
            with autocast, _live_lora(model, device):
                for i, t in enumerate(times):
                    mm.throw_exception_if_processing_interrupted()
                    t_model = torch.full((batch_size,), t * model.cfg.flow.num_train_timesteps, device=device)
                    v = dit(x.to(dtype), t_model, cond, y_mask=mask).x.float()
                    x0 = x - t * v
                    if i + 1 < len(times):
                        # predict the clean image, then re-noise it with fresh noise to the next time
                        fresh = torch.randn(x.shape, generator=generator, device=device)
                        x = (1.0 - times[i + 1]) * x0 + times[i + 1] * fresh
                    pbar.update_absolute(i + 1, steps, _preview(x0) if previews and i + 1 < steps else None)
            images = ((x0.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).float().cpu()
        finally:
            dit.to("cpu")
            _empty_cache()
        return (images,)


class IrisUpscaler:
    def __init__(self, model: IrisDiT, cfg, dtype: torch.dtype, settings: dict, prompt: dict):
        self.model, self.cfg, self.dtype = model, cfg, dtype
        self.sigma = float(settings["sigma"])
        self.tile = int(settings.get("tile", 1024))
        self.embeddings = prompt["embeddings"].to(dtype)
        self.mask = prompt["mask"].bool()
        self.nbytes = _nbytes(model)


def _task_prompt(weights: str, task_dir: str) -> str:
    """empty_prompt.safetensors next to Hub-layout weights, else the pack's cached copy
    of ``<task_dir>/empty_prompt.safetensors`` from the Hub."""
    side = os.path.join(os.path.dirname(weights), "empty_prompt.safetensors")
    if os.path.basename(weights) == "model.safetensors" and os.path.isfile(side):
        return side
    cached = os.path.join(NULL_CACHE_DIR, f"iris-3b-{task_dir}_empty_prompt.safetensors")
    return cached if os.path.isfile(cached) else _fetch(IRIS_REPO, f"{task_dir}/empty_prompt.safetensors", cached)


class IrisUpscalerLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (_with_default(_iris_files("upscaler"), UPSCALER_FILE), {
                    "tooltip": f"Iris upscaler weights in models/diffusion_models. {UPSCALER_FILE} is "
                               "downloaded on first use if missing (12 GB)",
                }),
                "weight_dtype": DTYPE_INPUT,
            }
        }

    RETURN_TYPES = ("IRIS_UPSCALER",)
    FUNCTION = "load"
    CATEGORY = "Iris-3B"

    @classmethod
    def VALIDATE_INPUTS(cls, model_name):
        return True  # a missing default is downloaded by load()

    def load(self, model_name, weight_dtype):
        weights = _weights_path(model_name)
        dtype = DTYPES[weight_dtype]
        model, cfg, task = _build_dit(weights, _config_path(weights, STOCK[UPSCALER_FILE][1]), dtype)
        if not task or task.get("name") != "restoration":
            raise ValueError(f"{model_name} is not an Iris restoration model")
        prompt = load_file(_task_prompt(weights, "upscaler"))
        log.info("loaded %s as %s", model_name, weight_dtype)
        return (IrisUpscaler(model, cfg, dtype, task, prompt),)


class IrisUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscaler": ("IRIS_UPSCALER",),
                "image": ("IMAGE",),
                "scale": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 4.0, "step": 0.5,
                                    "tooltip": "trained for 4x. Lower values restore at a smaller size"}),
                "limit_input_size": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "shrink the input first so its short side is at most 512 px and its long side "
                               "at most 1024 px (at 4x: at most 2048x4096, 21 tiles). Off keeps the input "
                               "size, and the time grows with the output area",
                }),
                "color_fix": ("BOOLEAN", {"default": True,
                                          "tooltip": "keep the input's colors (wavelet color fix)"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "Iris-3B"
    DESCRIPTION = "Iris-3B restoration and 4x upscaling. One model pass per 1024 px tile."

    @torch.no_grad()
    def upscale(self, upscaler, image, scale, limit_input_size, color_fix):
        from bucky_iris3b.downstream.restoration import gaussian_window, tile_positions, wavelet_color_fix

        F = torch.nn.functional
        device = mm.get_torch_device()
        up = upscaler
        tile, stride, patch = up.tile, up.tile // 2, up.cfg.model.patch_size
        mm.free_memory(up.nbytes + 2 * GB, device)
        up.model.to(device)
        embeddings = up.embeddings.to(device)
        mask = up.mask.to(device)
        t_model = torch.full((1,), up.sigma * up.cfg.flow.num_train_timesteps, device=device)

        def restore(x):
            mm.throw_exception_if_processing_interrupted()
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                v = up.model(x.to(up.dtype), t_model, embeddings, y_mask=mask).x
            return x.float() - up.sigma * v.float()

        outputs = []
        try:
            for frame in image:
                lq = frame[..., :3].permute(2, 0, 1)[None].to(device, torch.float32)
                if limit_input_size:
                    h, w = lq.shape[-2:]
                    shrink = min(1.0, 512 / min(h, w), 1024 / max(h, w))
                    if shrink < 1.0:
                        size = (max(1, round(h * shrink)), max(1, round(w * shrink)))
                        lq = F.interpolate(lq, size=size, mode="bicubic", align_corners=False, antialias=True)
                reference = F.interpolate(lq, scale_factor=scale, mode="bicubic", align_corners=False)
                out_size = reference.shape[-2:]
                x = reference
                if min(out_size) <= tile:
                    ratio = tile / min(out_size)
                    work = tuple(max(tile, round(side * ratio)) for side in out_size)
                    x = F.interpolate(x, size=work, mode="bicubic", align_corners=False, antialias=True)
                height, width = x.shape[-2:]
                x = F.pad(x * 2 - 1, (0, -width % patch, 0, -height % patch))

                # upstream's tiled(): 50% overlap, Gaussian-weighted fusion, plus a progress bar
                rows = tile_positions(x.shape[-2], tile, stride)
                cols = tile_positions(x.shape[-1], tile, stride)
                pbar = comfy.utils.ProgressBar(len(rows) * len(cols))
                if len(rows) == 1 and len(cols) == 1:
                    y = restore(x)
                    pbar.update_absolute(1)
                else:
                    window = gaussian_window(tile, device)
                    y = torch.zeros_like(x)
                    weight = torch.zeros_like(x[:, :1])
                    done = 0
                    for top in rows:
                        for left in cols:
                            crop = (..., slice(top, top + tile), slice(left, left + tile))
                            y[crop] += restore(x[crop].contiguous()) * window
                            weight[crop] += window
                            done += 1
                            pbar.update_absolute(done)
                    y = y / weight

                y = (y[..., :height, :width] + 1) / 2
                if (height, width) != tuple(out_size):
                    y = F.interpolate(y, size=out_size, mode="bicubic", align_corners=False, antialias=True)
                if color_fix:
                    y = wavelet_color_fix(y, reference)
                outputs.append(y.clamp(0, 1)[0].permute(1, 2, 0).cpu())
        finally:
            up.model.to("cpu")
            _empty_cache()
        if len({o.shape for o in outputs}) != 1:
            raise ValueError("images in the batch upscaled to different sizes")
        return (torch.stack(outputs),)


class IrisDepthModel:
    def __init__(self, model, cfg, dtype: torch.dtype, prompt: dict):
        self.model, self.cfg, self.dtype = model, cfg, dtype
        self.embeddings = prompt["embeddings"].to(dtype)
        self.mask = prompt["mask"].bool()
        self.nbytes = _nbytes(model)


class IrisDepthLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (_with_default(_iris_files("depth"), DEPTH_FILE), {
                    "tooltip": f"Iris depth weights in models/diffusion_models. {DEPTH_FILE} is "
                               "downloaded on first use if missing (12 GB)",
                }),
                "weight_dtype": DTYPE_INPUT,
            }
        }

    RETURN_TYPES = ("IRIS_DEPTH_MODEL",)
    FUNCTION = "load"
    CATEGORY = "Iris-3B"

    @classmethod
    def VALIDATE_INPUTS(cls, model_name):
        return True  # a missing default is downloaded by load()

    def load(self, model_name, weight_dtype):
        from bucky_iris3b.downstream.depth import IrisDepth

        weights_path = _weights_path(model_name)
        dtype = DTYPES[weight_dtype]
        raw = OmegaConf.to_container(OmegaConf.load(_config_path(weights_path, STOCK[DEPTH_FILE][1])))
        task = raw.pop("task", None)
        if not task or task.get("name") != "depth":
            raise ValueError(f"{model_name} is not an Iris depth model")
        cfg = inference_config(raw)
        weights = _load_weights(weights_path, dtype)
        with torch.device("meta"):
            model = IrisDepth(cfg.model)
        model.num_train_timesteps = cfg.flow.num_train_timesteps
        model.load_state_dict(weights, strict=True, assign=True)
        del weights
        model = model.train(False).requires_grad_(False).to(dtype=dtype)
        prompt = load_file(_task_prompt(weights_path, "depth"))
        log.info("loaded %s as %s", model_name, weight_dtype)
        return (IrisDepthModel(model, cfg, dtype, prompt),)


class IrisDepthEstimate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth_model": ("IRIS_DEPTH_MODEL",),
                "image": ("IMAGE",),
                "max_side": ("INT", {"default": 1024, "min": 0, "max": 4096, "step": 16,
                                     "tooltip": "long side the model sees; the map is resized back to the input "
                                                "size. 0 keeps the input size"}),
                "colors": (["grayscale (near = white)", "inferno"], {
                    "default": "grayscale (near = white)",
                    "tooltip": "grayscale is what depth ControlNets expect",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IRIS_DEPTH")
    RETURN_NAMES = ("depth", "depth_mask", "depth_3d")
    FUNCTION = "estimate"
    CATEGORY = "Iris-3B"
    DESCRIPTION = ("Iris-3B monocular depth: one model pass per image. Relative depth (not metric), "
                   "scaled to each image's 2nd to 98th percentile.")

    @torch.no_grad()
    def estimate(self, depth_model, image, max_side, colors):
        F = torch.nn.functional
        dm = depth_model
        device = mm.get_torch_device()
        patch = dm.cfg.model.patch_size
        mm.free_memory(dm.nbytes + 2 * GB, device)
        dm.model.to(device)
        embeddings, mask = dm.embeddings.to(device), dm.mask.to(device)
        maps, raws = [], []
        pbar = comfy.utils.ProgressBar(len(image))
        try:
            for i, frame in enumerate(image):
                mm.throw_exception_if_processing_interrupted()
                rgb = frame[..., :3].permute(2, 0, 1)[None].to(device, torch.float32)
                height, width = rgb.shape[-2:]
                scale = min(1.0, max_side / max(width, height)) if max_side else 1.0
                size = tuple(max(patch, round(side * scale / patch) * patch) for side in (height, width))
                if size != (height, width):
                    rgb = F.interpolate(rgb, size=size, mode="bicubic", align_corners=False, antialias=True)
                rgb = rgb.clamp(0, 1) * 2 - 1
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    depth = dm.model(rgb.to(dm.dtype), embeddings, mask).float()
                if depth.shape[-2:] != (height, width):
                    depth = F.interpolate(depth, size=(height, width), mode="bilinear", align_corners=False)
                # upstream colorize(): near = bright over the 2nd..98th percentile
                d = depth[0, 0]
                raws.append(d.cpu())
                low, high = torch.quantile(d.flatten()[:: max(1, d.numel() // 1_000_000)], torch.tensor([0.02, 0.98], device=device))
                maps.append(((high - d) / (high - low).clamp_min(1e-6)).clamp(0, 1).cpu())
                pbar.update_absolute(i + 1)
        finally:
            dm.model.to("cpu")
            _empty_cache()
        near = torch.stack(maps)
        if colors == "inferno":
            try:
                from matplotlib import colormaps

                rgb = torch.from_numpy(colormaps["inferno"](near.numpy())[..., :3]).float()
            except ImportError:
                log.warning("matplotlib is missing, using grayscale")
                rgb = near[..., None].expand(-1, -1, -1, 3)
        else:
            rgb = near[..., None].expand(-1, -1, -1, 3).contiguous()
        return (rgb, near, {"log_depth": raws, "image": image[..., :3].cpu()})


def _write_ply(path: str, xyz, rgb, faces=None) -> None:
    """Binary little-endian PLY: float xyz, uchar rgb, optional triangle faces."""
    import numpy as np

    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(xyz)}",
              "property float x", "property float y", "property float z",
              "property uchar red", "property uchar green", "property uchar blue"]
    if faces is not None:
        header += [f"element face {len(faces)}", "property list uchar int vertex_indices"]
    header.append("end_header")
    vertex = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                       ("r", "u1"), ("g", "u1"), ("b", "u1")])
    vertex["x"], vertex["y"], vertex["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    vertex["r"], vertex["g"], vertex["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(vertex.tobytes())
        if faces is not None:
            face = np.empty(len(faces), dtype=[("n", "u1"), ("i", "<i4", (3,))])
            face["n"], face["i"] = 3, faces
            f.write(face.tobytes())


class IrisDepthTo3D:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth_3d": ("IRIS_DEPTH", {"tooltip": "the depth_3d output of Iris Depth"}),
                "mode": (["point cloud", "mesh"], {
                    "default": "point cloud",
                    "tooltip": "mesh joins neighboring points into triangles and leaves gaps at depth edges",
                }),
                "fov": ("FLOAT", {"default": 60.0, "min": 10.0, "max": 150.0, "step": 1.0,
                                  "tooltip": "assumed horizontal field of view of the photo, in degrees"}),
                "max_points": ("INT", {"default": 500000, "min": 10000, "max": 4000000, "step": 10000,
                                       "tooltip": "the depth map is subsampled to stay under this"}),
                "filename_prefix": ("STRING", {"default": "iris3b_depth3d"}),
            }
        }

    RETURN_TYPES = ("FILE_3D",)
    RETURN_NAMES = ("model_3d",)
    FUNCTION = "build"
    OUTPUT_NODE = True
    CATEGORY = "Iris-3B"
    DESCRIPTION = ("Turns Iris depth into a colored point cloud or mesh (.ply in the output folder). "
                   "Connect it to Preview 3D to orbit around it.")

    def build(self, depth_3d, mode, fov, max_points, filename_prefix):
        import numpy as np

        log_depth, image = depth_3d["log_depth"], depth_3d["image"]
        out_dir = folder_paths.get_output_directory()
        full, name, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, out_dir)
        paths = []
        for i, d in enumerate(log_depth):
            d = d.float().numpy()
            h, w = d.shape
            rgb = (image[i].float().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
            stride = max(1, int(np.ceil(np.sqrt(h * w / max_points))))
            v, u = np.mgrid[0:h:stride, 0:w:stride]
            # as in upstream's demo viewer: exp of relative log depth is depth up to
            # scale, rescaled to a median of 3 for viewing
            z = np.exp(d[v, u].astype(np.float64))
            z = 3 * z / np.median(z)
            f = (w / 2) / np.tan(np.radians(fov) / 2)
            # three.js is y-up and looks down -z
            xyz = np.stack([(u + 0.5 - w / 2) * z / f, -(v + 0.5 - h / 2) * z / f, -z], -1).reshape(-1, 3)
            colors = rgb[v, u].reshape(-1, 3)
            faces = None
            if mode == "mesh":
                gh, gw = v.shape
                idx = np.arange(gh * gw).reshape(gh, gw)
                a, b, c, e = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
                tris = np.concatenate([np.stack([a, c, b], -1).reshape(-1, 3),
                                       np.stack([b, c, e], -1).reshape(-1, 3)])
                zf = z.reshape(-1)[tris]
                # drop triangles that stretch across a depth edge
                tris = tris[zf.max(1) / zf.min(1) < 1.08]
                faces = tris.astype(np.int32)
            path = os.path.join(full, f"{name}_{counter + i:05}_.ply")
            _write_ply(path, xyz.astype(np.float32), colors, faces)
            paths.append(path)
        log.info("wrote %s", ", ".join(paths))
        try:
            from comfy_api.latest import Types

            model = Types.File3D(paths[0])
        except Exception:
            model = paths[0]
        return (model,)


def _previews_enabled() -> bool:
    try:
        from comfy.cli_args import LatentPreviewMethod, args
        return args.preview_method != LatentPreviewMethod.NoPreviews
    except Exception:
        return True


def _preview(x0: torch.Tensor):
    from PIL import Image

    try:
        from latent_preview import MAX_PREVIEW_RESOLUTION
    except Exception:
        MAX_PREVIEW_RESOLUTION = 512
    img = x0[:1].float().clamp(-1, 1)
    scale = MAX_PREVIEW_RESOLUTION / max(img.shape[-2:])
    if scale < 1:
        img = torch.nn.functional.interpolate(img, scale_factor=scale, mode="area")
    arr = ((img[0] + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    return ("JPEG", Image.fromarray(arr), MAX_PREVIEW_RESOLUTION)


NODE_CLASS_MAPPINGS = {
    "IrisModelLoader": IrisModelLoader,
    "IrisTextEncoderLoader": IrisTextEncoderLoader,
    "IrisTextEncode": IrisTextEncode,
    "IrisSamplerSimple": IrisSamplerSimple,
    "IrisSamplerAdvanced": IrisSamplerAdvanced,
    "IrisLoRA": IrisLoRA,
    "IrisTurboLoRA": IrisTurboLoRA,
    "IrisSamplerTurbo": IrisSamplerTurbo,
    "IrisUpscalerLoader": IrisUpscalerLoader,
    "IrisUpscale": IrisUpscale,
    "IrisDepthLoader": IrisDepthLoader,
    "IrisDepthEstimate": IrisDepthEstimate,
    "IrisDepthTo3D": IrisDepthTo3D,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IrisModelLoader": "Load Iris-3B Model",
    "IrisTextEncoderLoader": "Load Iris Text Encoder (Qwen3-VL-4B)",
    "IrisTextEncode": "Iris Text Encode",
    "IrisSamplerSimple": "Iris Sampler (Simple)",
    "IrisSamplerAdvanced": "Iris Sampler (Advanced)",
    "IrisLoRA": "Load Iris LoRA",
    "IrisTurboLoRA": "Load Iris Turbo LoRA",
    "IrisSamplerTurbo": "Iris Sampler (Turbo)",
    "IrisUpscalerLoader": "Load Iris Upscaler",
    "IrisUpscale": "Iris Upscale",
    "IrisDepthLoader": "Load Iris Depth Model",
    "IrisDepthEstimate": "Iris Depth",
    "IrisDepthTo3D": "Iris Depth to 3D",
}
