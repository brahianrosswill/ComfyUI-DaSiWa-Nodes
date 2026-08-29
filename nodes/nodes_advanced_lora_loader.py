"""Advanced LoRA Loader - universal LoRA stacker.

The loader supports ordinary image/video LoRAs plus LTX-2.3's separate video
and audio branches. Each of the 10 slots has:
  - lora_str: master LoRA strength (-5.0 to 5.0; applied to both branches)
  - vs: video branch multiplier (0.0-2.0, default 1.0)
  - as: audio branch multiplier (0.0-2.0, default 1.0)

Effective video strength = lora_str * vs
Effective audio strength = lora_str * as
"""

import os
import json
import folder_paths
import comfy.utils
import comfy.lora
try:
    from comfy.lora import load_lora_for_models as _load_lora
except (ImportError, AttributeError):
    from comfy.sd import load_lora_for_models as _load_lora
from aiohttp import web
from server import PromptServer
try:
    from .helper_logging import log_dasiwa
except ImportError:
    from helper_logging import log_dasiwa

NUM_SLOTS = 10
MODEL_TYPE_BASIC = "Basic"
MODEL_TYPE_LTX23 = "LTX-2.3"
MODEL_TYPES = (MODEL_TYPE_BASIC, MODEL_TYPE_LTX23)


def _is_audio_key(k):
    """Keys containing 'audio' = audio branch"""
    return "audio" in k.lower()


def _normalize_model_type(model_type):
    return model_type if model_type in MODEL_TYPES else MODEL_TYPE_BASIC


def _apply_full_lora(model, clip, weights, strength):
    if weights and strength != 0.0:
        return _load_lora(model, clip, weights, strength, strength)
    return model, clip


def _apply_slot(model, clip, lora_name, lora_str, vs, as_, model_type):
    """Apply a single LoRA slot for the selected model architecture."""
    lora_path = folder_paths.get_full_path("loras", lora_name)
    if not lora_path or not os.path.isfile(lora_path):
        log_dasiwa("Advanced LoRA Loader", f"LoRA not found: {lora_name}")
        return model, clip

    weights = comfy.utils.load_torch_file(lora_path, safe_load=True)
    v_final = lora_str * vs
    model_type = _normalize_model_type(model_type)
    if model_type != MODEL_TYPE_LTX23:
        log_dasiwa(
            "Advanced LoRA Loader",
            f"'{lora_name}' mode={model_type} full:{len(weights)}@{v_final:.2f}",
        )
        return _apply_full_lora(model, clip, weights, v_final)

    video_weights = {k: v for k, v in weights.items() if not _is_audio_key(k)}
    audio_weights = {k: v for k, v in weights.items() if _is_audio_key(k)}
    a_final = lora_str * as_
    log_dasiwa(
        "Advanced LoRA Loader",
        f"'{lora_name}' mode=LTX-2.3 V:{len(video_weights)}@{v_final:.2f} A:{len(audio_weights)}@{a_final:.2f}",
    )
    model, clip = _apply_full_lora(model, clip, video_weights, v_final)
    return _apply_full_lora(model, clip, audio_weights, a_final)


# ── Key count endpoint ────────────────────────────────────────────────────────
@PromptServer.instance.routes.get("/dasiwa/ltx2/keycounts")
async def keycounts(request):
    """Return full-map or LTX-2.3 video/audio LoRA tensor counts."""
    lora_name = request.rel_url.query.get("lora", "")
    model_type = _normalize_model_type(request.rel_url.query.get("model_type", MODEL_TYPE_LTX23))
    if not lora_name:
        return web.json_response({"v": 0, "a": 0, "mode": model_type})
    lora_path = folder_paths.get_full_path("loras", lora_name)
    if not lora_path or not os.path.isfile(lora_path):
        return web.json_response({"v": 0, "a": 0, "mode": model_type})
    try:
        import safetensors
        with safetensors.safe_open(lora_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
    except Exception:
        try:
            weights = comfy.utils.load_torch_file(lora_path, safe_load=True)
            keys = list(weights.keys())
        except Exception:
            return web.json_response({"v": -1, "a": -1, "mode": model_type})
    if model_type != MODEL_TYPE_LTX23:
        return web.json_response({"v": len(keys), "a": 0, "mode": model_type})
    v = sum(1 for k in keys if not _is_audio_key(k))
    a = sum(1 for k in keys if _is_audio_key(k))
    return web.json_response({"v": v, "a": a, "mode": model_type})


# ── Node ──────────────────────────────────────────────────────────────────────
class DaSiWa_AdvancedLoRALoader:
    """
    Advanced LoRA Loader

    10-slot LoRA stacker for ordinary image/video LoRAs and LTX-2.3 workflows.
    """
    DESCRIPTION = (
        "Advanced LoRA Loader: stacks ordinary image/video LoRAs or LTX-2.3 "
        "video/audio LoRAs. Audio separation is available only for LTX-2.3."
    )

    @classmethod
    def INPUT_TYPES(cls):
        lora_list = ["None"] + folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL", {"description": "Base model that will receive the active LoRA stack."}),
                "clip": ("CLIP", {"description": "CLIP/text encoder paired with the model; LoRA weights are applied when compatible."}),
                "stack_data": ("STRING", {"default": "[]", "multiline": False, "description": "JSON-encoded LoRA slot data managed by the custom UI. Each slot stores on/off, LoRA file, master strength, video multiplier, and audio multiplier."}),
                "model_type": (["Basic", "LTX-2.3"], {
                    "default": "Basic",
                    "description": "Basic loads all LoRA tensors universally; LTX-2.3 enables video/audio separation.",
                }),
            },
            "hidden": {"available_loras": (lora_list, {"description": "Internal list of LoRA files used by the custom slot picker."})}
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "apply_stack"
    CATEGORY = "DaSiWa/loaders/lora"

    def apply_stack(self, model, clip, stack_data="[]", model_type=MODEL_TYPE_BASIC, available_loras=None):
        """Parse and apply the LoRA stack"""
        m, c = model, clip
        try:
            data = json.loads(stack_data)
        except Exception as e:
            log_dasiwa("Advanced LoRA Loader", f"Failed to parse stack_data: {e}")
            return (m, c)
        
        for i, row in enumerate(data):
            if not row.get("on") or row.get("lora") in ("None", "", None):
                continue
            lora_str = float(row.get("str", 1.0))
            vs = float(row.get("vs", 1.0))
            as_ = float(row.get("as", 1.0))
            m, c = _apply_slot(m, c, row["lora"], lora_str, vs, as_, model_type)
        
        return (m, c)
