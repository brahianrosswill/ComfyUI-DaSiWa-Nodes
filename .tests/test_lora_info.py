import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "nodes" / "lora_info.py"


@pytest.fixture
def info_module(monkeypatch):
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda _category, name: name

    aiohttp = types.ModuleType("aiohttp")
    aiohttp.web = types.SimpleNamespace(json_response=lambda payload: payload)

    server = types.ModuleType("server")
    server.PromptServer = types.SimpleNamespace(
        instance=types.SimpleNamespace(routes=types.SimpleNamespace(get=lambda _path: (lambda handler: handler)))
    )
    helper_logging = types.ModuleType("helper_logging")
    helper_logging.log_dasiwa = lambda *_args, **_kwargs: None

    for name, module in {
        "folder_paths": folder_paths,
        "aiohttp": aiohttp,
        "server": server,
        "helper_logging": helper_logging,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("lora_info_under_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sha256_file_chunks_the_whole_file(info_module, tmp_path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"abc" * 2000)
    assert info_module.sha256_file(str(path)) == hashlib.sha256(b"abc" * 2000).hexdigest()


def test_header_metadata_reads_safetensors_header(info_module, tmp_path):
    metadata = {"ss_tag_frequency": '{"A": {"sks:045": 1, "sks:090": 2}, "B": {"sks:045": 3}}',
                "ss_output_name": "Cool LoRA"}
    header = json.dumps({"__metadata__": metadata}).encode()
    path = tmp_path / "m.safetensors"
    path.write_bytes(len(header).to_bytes(8, "little") + header + b"xx")
    md = info_module.header_metadata(str(path))
    assert md["ss_output_name"] == "Cool LoRA"
    # ss_tag_frequency ships as a JSON string; the helper parses it in place.
    freq = md["ss_tag_frequency"]
    assert freq["A"]["sks:045"] == 1 and freq["B"]["sks:045"] == 3


def test_header_metadata_missing_returns_empty(info_module, tmp_path):
    path = tmp_path / "bad.bin"
    path.write_bytes(b"12345678" + b"x" * 8)
    assert info_module.header_metadata(str(path)) == {}


def test_header_metadata_missing_file_returns_empty(info_module, tmp_path):
    assert info_module.header_metadata(str(tmp_path / "nope.safetensors")) == {}


def test_trained_words_aggregates_buckets(info_module):
    md = {"ss_tag_frequency": json.dumps({"A": {"w1": 2, "w2": 1}, "B": {"w1": 3}})}
    words = info_module.trained_words_from_metadata(md)
    by = {w["word"]: w for w in words}
    assert by["w1"]["count"] == 5 and by["w2"]["count"] == 1


def test_trained_words_tolerates_non_numeric_and_junk(info_module):
    md = {"ss_tag_frequency": json.dumps({"A": {"w1": "??", "w2": 4}, "junk": "not-a-dict"})}
    words = {w["word"]: w for w in info_module.trained_words_from_metadata(md)}
    assert words["w1"]["count"] == 0 and words["w2"]["count"] == 4


def test_trained_words_no_metadata_returns_empty(info_module):
    assert info_module.trained_words_from_metadata({}) == []
