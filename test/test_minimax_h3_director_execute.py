"""Contract tests for Director-owned Image Inpaint execution."""
from nodes.helper_minimax_h3_director_execute import (
    DEFAULT_POSTPROCESS_RECIPE,
    normalize_postprocess_recipe,
)


def test_recipe_defaults_are_disabled_and_stably_ordered():
    recipe = normalize_postprocess_recipe(None)

    assert recipe == DEFAULT_POSTPROCESS_RECIPE
    assert [stage["id"] for stage in recipe] == [
        "frame_interpolation", "torch_resize", "model_upscale", "rtx_refiner", "watermark",
    ]
    assert not any(stage["enabled"] for stage in recipe)


def test_recipe_normalization_keeps_known_overrides_and_ignores_unknown_stages():
    recipe = normalize_postprocess_recipe([
        {"id": "torch_resize", "enabled": True, "scale_multiplier": 3},
        {"id": "unknown", "enabled": True},
    ])

    resize = next(stage for stage in recipe if stage["id"] == "torch_resize")
    assert resize["enabled"] is True
    assert resize["scale_multiplier"] == 3
    assert len(recipe) == len(DEFAULT_POSTPROCESS_RECIPE)
