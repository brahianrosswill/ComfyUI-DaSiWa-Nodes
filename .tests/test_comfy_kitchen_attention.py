import importlib.util
import sys
import types
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "nodes" / "nodes_comfy_kitchen_attention.py"
PACKAGE_PATH = Path(__file__).parents[1] / "__init__.py"


@pytest.fixture
def kitchen_module(monkeypatch):
    attention = types.ModuleType("comfy.ldm.modules.attention")
    attention.COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE = True
    attention.get_attention_function = lambda _name, default=None: default
    attention.optimized_attention = object()
    helper_logging = types.ModuleType("helper_logging")
    helper_logging.log_dasiwa = lambda *_args: None

    comfy = types.ModuleType("comfy")
    ldm = types.ModuleType("comfy.ldm")
    modules = types.ModuleType("comfy.ldm.modules")
    comfy.ldm = ldm
    ldm.modules = modules
    modules.attention = attention

    for name, module in {
        "comfy": comfy,
        "comfy.ldm": ldm,
        "comfy.ldm.modules": modules,
        "comfy.ldm.modules.attention": attention,
        "helper_logging": helper_logging,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("nodes_comfy_kitchen_attention_under_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_schema_uses_requested_node_name_and_dasiwa_category(kitchen_module):
    node = kitchen_module.PathchComfyKitchenAttentionDaSiWa

    assert node.INPUT_TYPES() == {"required": {"model": ("MODEL",)}}
    assert node.RETURN_TYPES == ("MODEL",)
    assert node.FUNCTION == "patch"
    assert node.CATEGORY == "DaSiWa"


def test_patch_clones_model_and_assigns_registered_backend(kitchen_module):
    backend = object()
    kitchen_module.attention.get_attention_function = (
        lambda name, default=None: backend if name == "comfy_kitchen_int8" else default
    )

    class ModelClone:
        def __init__(self):
            self.backend = None

        def set_model_optimized_attention(self, selected_backend):
            self.backend = selected_backend

    class Model:
        def __init__(self):
            self.clone_calls = 0
            self.result = ModelClone()

        def clone(self):
            self.clone_calls += 1
            return self.result

    model = Model()

    assert kitchen_module.PathchComfyKitchenAttentionDaSiWa().patch(model) == (model.result,)
    assert model.clone_calls == 1
    assert model.result.backend is backend


def test_patch_reports_comfy_kitchen_selection(kitchen_module, monkeypatch):
    backend = object()
    messages = []
    kitchen_module.attention.get_attention_function = lambda _name, _default=None: backend
    monkeypatch.setattr(kitchen_module, "log_dasiwa", lambda component, message: messages.append((component, message)))

    class Model:
        def clone(self):
            return self

        def set_model_optimized_attention(self, selected_backend):
            assert selected_backend is backend

    kitchen_module.PathchComfyKitchenAttentionDaSiWa().patch(Model())

    assert messages == [("Patch Comfy Kitchen Attention", "Using Comfy Kitchen INT8 attention.")]


def test_patch_falls_back_to_comfyui_default_attention_and_reports_it(kitchen_module, monkeypatch):
    default_backend = object()
    messages = []
    kitchen_module.attention.COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE = False
    kitchen_module.attention.optimized_attention = default_backend
    monkeypatch.setattr(kitchen_module, "log_dasiwa", lambda component, message: messages.append((component, message)))

    class Model:
        def clone(self):
            return self

        def set_model_optimized_attention(self, selected_backend):
            self.backend = selected_backend

    model = Model()
    assert kitchen_module.PathchComfyKitchenAttentionDaSiWa().patch(model) == (model,)
    assert model.backend is default_backend
    assert messages == [
        ("Patch Comfy Kitchen Attention", "Comfy Kitchen INT8 attention is unavailable; using ComfyUI default attention.")
    ]


def test_patch_rejects_old_model_patcher_api(kitchen_module):
    kitchen_module.attention.get_attention_function = lambda _name, _default=None: object()

    class Model:
        def clone(self):
            return object()

    with pytest.raises(RuntimeError, match="lacks set_model_optimized_attention"):
        kitchen_module.PathchComfyKitchenAttentionDaSiWa().patch(Model())


def test_package_registers_requested_class_and_display_name():
    package_source = PACKAGE_PATH.read_text(encoding="utf-8")

    assert "PathchComfyKitchenAttentionDaSiWa" in package_source
    assert '"Patch Comfy Kitchen Attention"' in package_source
