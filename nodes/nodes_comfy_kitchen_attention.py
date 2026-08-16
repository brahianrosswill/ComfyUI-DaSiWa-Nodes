import comfy.ldm.modules.attention as attention

try:
    from .helper_logging import log_dasiwa
except ImportError:
    from helper_logging import log_dasiwa


class PathchComfyKitchenAttentionDaSiWa:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "DaSiWa"
    DESCRIPTION = "Patches this model to use Comfy Kitchen INT8 attention at runtime."

    def patch(self, model):
        backend = None
        if getattr(attention, "COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE", False):
            backend = attention.get_attention_function("comfy_kitchen_int8", None)

        if backend is None:
            backend = attention.optimized_attention
            log_dasiwa(
                "Patch Comfy Kitchen Attention",
                "Comfy Kitchen INT8 attention is unavailable; using ComfyUI default attention.",
            )
        else:
            log_dasiwa("Patch Comfy Kitchen Attention", "Using Comfy Kitchen INT8 attention.")

        patched_model = model.clone()
        if not hasattr(patched_model, "set_model_optimized_attention"):
            raise RuntimeError(
                "This ComfyUI ModelPatcher lacks set_model_optimized_attention(). Update ComfyUI."
            )
        patched_model.set_model_optimized_attention(backend)
        return (patched_model,)
