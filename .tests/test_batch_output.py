import gc
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch


MODULE_PATH = Path(__file__).parents[1] / "nodes" / "helper_batch_output.py"
spec = importlib.util.spec_from_file_location("helper_batch_output", MODULE_PATH)
assert spec is not None and spec.loader is not None
batch_output = importlib.util.module_from_spec(spec)
sys.modules["helper_batch_output"] = batch_output
spec.loader.exec_module(batch_output)

LOGGING_PATH = Path(__file__).parents[1] / "nodes" / "helper_logging.py"
logging_spec = importlib.util.spec_from_file_location("helper_logging", LOGGING_PATH)
assert logging_spec is not None and logging_spec.loader is not None
helper_logging = importlib.util.module_from_spec(logging_spec)
sys.modules["helper_logging"] = helper_logging
logging_spec.loader.exec_module(helper_logging)


def test_log_dasiwa_writes_a_prefixed_comfyui_console_message(capsys):
    helper_logging.log_dasiwa("Unit Test", "completed work.")

    assert capsys.readouterr().out == "\033[38;5;136m[DaSiWa Unit Test]\033[0m completed work.\n"


def test_log_startup_summary_reports_loaded_nodes_and_the_important_terms(capsys):
    helper_logging.log_startup_summary(17)

    messages = capsys.readouterr().out.splitlines()
    assert len(messages) == 2
    assert "[DaSiWa Nodes]" in messages[0]
    assert "Loaded 17 extraordinarily overengineered nodes. 🐈" in messages[0]
    assert "cat ears improve everything" in messages[1]
    assert "SlimeGirls deserve rights" in messages[1]
    assert "Dragoniods need more screen time" in messages[1]
    assert "darkness is the correct light source" in messages[1]


def test_uses_available_ram_not_a_fixed_output_cap(monkeypatch, tmp_path):
    monkeypatch.setattr(batch_output, "total_ram_bytes", lambda: 128 * 1024 ** 3)
    monkeypatch.setattr(
        batch_output, "available_ram_bytes", lambda: batch_output.ram_safety_reserve_bytes() + 16
    )

    output, storage_path = batch_output.allocate_cpu_output((1, 2, 2, 1), torch.float32, str(tmp_path))

    assert storage_path is None
    assert output.device.type == "cpu"


def test_uses_mmap_when_available_ram_would_cross_reserve(monkeypatch, tmp_path):
    monkeypatch.setattr(batch_output, "total_ram_bytes", lambda: 128 * 1024 ** 3)
    monkeypatch.setattr(batch_output, "available_ram_bytes", lambda: batch_output.ram_safety_reserve_bytes())

    output, storage_path = batch_output.allocate_cpu_output((1, 2, 2, 1), torch.float32, str(tmp_path))

    assert storage_path is not None
    assert os.path.exists(storage_path)
    del output
    gc.collect()
    assert not os.path.exists(storage_path)


def test_low_ram_system_uses_a_proportional_reserve(monkeypatch):
    monkeypatch.setattr(batch_output, "total_ram_bytes", lambda: 8 * 1024 ** 3)

    assert batch_output.ram_safety_reserve_bytes() == 2 * 1024 ** 3


def test_reserve_is_capped_at_eight_gib_on_large_memory_systems(monkeypatch):
    monkeypatch.setattr(batch_output, "total_ram_bytes", lambda: 128 * 1024 ** 3)

    assert batch_output.ram_safety_reserve_bytes() == 8 * 1024 ** 3


def test_unload_all_comfy_models_without_model_management_returns_false(monkeypatch):
    import builtins
    import sys
    monkeypatch.delitem(sys.modules, "model_management", raising=False)

    real_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name == "model_management":
            raise ImportError("no model_management in plain pytest")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)

    assert batch_output.unload_all_comfy_models() is False


def test_unload_all_comfy_models_calls_unload_and_soft_empty_cache(monkeypatch):
    import sys
    import types
    calls = []
    mm = types.ModuleType("model_management")
    mm.unload_all_models = lambda: calls.append("unload_all_models")
    mm.soft_empty_cache = lambda: calls.append("soft_empty_cache")
    monkeypatch.setitem(sys.modules, "model_management", mm)

    assert batch_output.unload_all_comfy_models() is True
    assert calls == ["unload_all_models", "soft_empty_cache"]


def test_force_mmap_allocates_disk_backed_even_when_ram_is_available(tmp_path, monkeypatch):
    monkeypatch.setattr(batch_output, "can_allocate_in_ram", lambda _: True)

    output, storage_path = batch_output.allocate_cpu_output((1, 2, 2, 1), torch.float32, str(tmp_path), force_mmap=True)

    assert storage_path is not None
    assert os.path.exists(storage_path)
    assert output.device.type == "cpu"
    del output
    gc.collect()
    assert not os.path.exists(storage_path)


def test_force_mmap_without_ram_shortage_still_checks_disk_space(tmp_path, monkeypatch):
    monkeypatch.setattr(batch_output, "can_allocate_in_ram", lambda _: True)

    with pytest.raises(RuntimeError, match="Not enough free disk space"):
        batch_output.allocate_cpu_output(
            (1, 2, 2, 1), torch.float32, str(tmp_path),
            has_free_disk_space=lambda *_: False, force_mmap=True,
        )


def test_before_mmap_hook_runs_only_when_mmap_path_is_taken(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(batch_output, "can_allocate_in_ram", lambda _: True)

    # RAM fits and mmap not forced -> no mmap, hook must NOT run.
    _, path = batch_output.allocate_cpu_output(
        (1, 2, 2, 1), torch.float32, str(tmp_path),
        before_mmap=lambda: calls.append("ram"))
    assert path is None
    assert calls == []

    # Forced -> hook runs exactly once and before the temp file exists.
    seen = []

    def hook():
        seen.append(os.listdir(str(tmp_path)))

    output, path2 = batch_output.allocate_cpu_output(
        (1, 2, 2, 1), torch.float32, str(tmp_path), force_mmap=True, before_mmap=hook)
    assert path2 is not None
    assert len(seen) == 1
    assert os.path.basename(path2) not in seen[0]
    del output
    gc.collect()