"""MiniMax H3 Director Guide 2.0: thin adapter for ComfyUI's native MiniMax H3 nodes.

Standalone v2 fork of the Director Guide. It depends only on the v2 Director
helper (helper_minimax_h3_director_v2) so it never couples to the v1 Director.
"""

from .helper_minimax_h3_director_v2 import normalize_guide
from .helper_logging import log_dasiwa


def _describe_output(value) -> str:
    if isinstance(value, dict):
        details = []
        for key, item in value.items():
            shape = getattr(item, "shape", None)
            details.append(f"{key}:{tuple(shape) if shape is not None else type(item).__name__}")
        return "{" + ", ".join(details) + "}"
    shape = getattr(value, "shape", None)
    if shape is not None:
        return f"{type(value).__name__}{tuple(shape)}"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}[{len(value)}]"
    return type(value).__name__


def _native_node(name):
    """Resolve at execution so installed ComfyUI updates are used automatically."""
    try:
        from comfy_extras import nodes_minimax_h3
        return getattr(nodes_minimax_h3, name)
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            f"MiniMax H3 Director requires ComfyUI's native {name} node. "
            "Update ComfyUI to a version that includes MiniMax H3 support."
        ) from exc


class MiniMaxH3DirectorGuideV2:
    """Turn one Director 2.0 guide socket into the exact native MiniMax H3 call."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "guide": ("MINIMAX_H3_DIRECTOR_GUIDE",),
            },
            "optional": {"audio_vae": ("VAE",)},
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "apply"
    CATEGORY = "DaSiWa/MiniMax H3"

    def apply(self, clip, vae, guide, audio_vae=None):
        state = normalize_guide(guide)
        if state.mode in {"T2VA", "I2VA", "FL2VA", "L2VA", "Image Inpaint"}:
            native = _native_node("MiniMaxH3ImageToVideo")
            native_length = 5 if state.mode == "Image Inpaint" else state.length
            log_dasiwa(
                "MiniMax H3 Director Guide 2.0",
                f"mode={state.mode}; upstream=MiniMaxH3ImageToVideo; clip={type(clip).__name__}; "
                f"vae={type(vae).__name__}; audio_vae=not-used; frames={native_length}; "
                f"first_frame={state.first_frame is not None}; last_frame={state.last_frame is not None}",
            )
            positive, latent = native.execute(
                clip, vae, state.resolved_prompt, state.width, state.height, native_length,
                state.first_frame, state.last_frame,
            )
            log_dasiwa(
                "MiniMax H3 Director Guide 2.0",
                f"passed forward from MiniMaxH3ImageToVideo: conditioning={_describe_output(positive)}; "
                f"latent={_describe_output(latent)}",
            )
            return positive, latent

        if audio_vae is None:
            raise ValueError("audio_vae is required for REF2VA")
        native = _native_node("MiniMaxH3ReferenceToVideo")
        log_dasiwa(
            "MiniMax H3 Director Guide 2.0",
            f"mode=REF2VA; upstream=MiniMaxH3ReferenceToVideo; clip={type(clip).__name__}; "
            f"vae={type(vae).__name__}; audio_vae={type(audio_vae).__name__}; frames={state.length}; "
            f"refs=images:{len(state.ref_images)},videos:{len(state.ref_videos)},"
            f"video_audio:{len(state.ref_video_audios)},audio:{len(state.ref_audios)}",
        )
        positive, latent = native.execute(
            clip, vae, audio_vae, state.resolved_prompt, state.width, state.height, state.length,
            state.ref_image_size, state.ref_images, state.ref_videos,
            state.ref_video_audios, state.ref_audios,
        )
        log_dasiwa(
            "MiniMax H3 Director Guide 2.0",
            f"passed forward from MiniMaxH3ReferenceToVideo: conditioning={_describe_output(positive)}; "
            f"latent={_describe_output(latent)}",
        )
        return positive, latent


NODE_CLASS_MAPPINGS = {"MiniMaxH3DirectorGuideV2": MiniMaxH3DirectorGuideV2}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3DirectorGuideV2": "MiniMax H3 Director Guide 2.0"}
