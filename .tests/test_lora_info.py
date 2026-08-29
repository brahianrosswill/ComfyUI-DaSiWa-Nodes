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


CANNED_CIVITAI = {
    "id": 456,
    "name": "Version B",
    "type": "LoRA",
    "baseModel": "SDXL",
    "modelId": 123,
    "triggerWords": ["sks"],
    "trainedWords": ["sks", "myword"],
    "images": [
        {
            "url": "https://civitai.com/image/1.png",
            "type": "image",
            "width": 512,
            "height": 512,
            "meta": {"seed": 7, "prompt": "p", "negativePrompt": "n",
                     "steps": 20, "sampler": "EulerA", "cfgScale": 7, "Model": "SDXL"},
        },
        {"url": "https://civitai.com/image/2.mp4", "type": "video", "width": 512, "height": 512, "meta": {}},
    ],
}


def test_merge_civitai_words_images_link(info_module):
    info = {"trainedWords": [{"word": "w1", "count": 5}]}
    changed = info_module.merge_civitai(info, CANNED_CIVITAI)
    assert changed
    by = {w["word"]: w for w in info["trainedWords"]}
    assert by["sks"]["civitai"] is True
    assert by["myword"]["civitai"] is True
    assert by["w1"]["count"] == 5
    assert info["name"] == "Version B"
    assert info["type"] == "LoRA"
    assert info["baseModel"] == "SDXL"
    assert info["links"] == ["https://civitai.com/models/123?modelVersionId=456"]
    assert info["images"][0]["url"].endswith("1.png") and info["images"][0]["seed"] == 7
    assert info["images"][1]["type"] == "video"
    assert info["trainedWords"][0]["word"] == "w1"  # sorted desc by count


def test_merge_civitai_falls_back_to_model_object(info_module):
    civ = {"id": 1, "name": "V", "model": {"id": 9, "name": "Model A", "type": "LoRA", "baseModel": "SD1.5"},
           "modelId": 9, "images": []}
    info = {}
    info_module.merge_civitai(info, civ)
    assert info["name"] == "Model A - V"
    assert info["type"] == "LoRA"
    assert info["baseModel"] == "SD1.5"


def test_merge_civitai_none_is_noop(info_module):
    info = {}
    assert info_module.merge_civitai(info, None) is False
    assert info == {}


def test_cache_roundtrip(info_module, tmp_path, monkeypatch):
    monkeypatch.setattr(info_module, "CACHE_DIR", str(tmp_path / "cache"))
    info_module.cache_write("deadbeef", {"a": 1})
    assert info_module.cache_read("deadbeef") == {"a": 1}
    assert info_module.cache_read("missing") is None


def test_cache_write_never_raises_when_unwritable(info_module, tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    monkeypatch.setattr(info_module, "CACHE_DIR", str(blocker / "lorainfo"))
    info_module.cache_write("x", {"a": 1})  # makedirs under a file raises; must be swallowed
