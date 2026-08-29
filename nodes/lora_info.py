"""LoRA info backend for the Advanced LoRA Loader's "info" button.

Ports the rgthree Power Lora Loader info feature, scoped to this nodepack:
sha256 of the LoRA file, safetensors header metadata (trigger words), and a
Civitai lookup by sha256, cached next to the nodepack in lorainfo/<sha256>.json.
"""

import hashlib
import json
import os

CHUNK_SIZE = 128 * 1024
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lorainfo")


def sha256_file(path: str) -> str:
    """Chunked sha256 of a (possibly large) LoRA file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(block)
    return h.hexdigest()


def header_metadata(path: str) -> dict:
    """Read safetensors __metadata__ from the file header without torch.

    Returns {} for non-safetensors / unreadable files. String values that are
    themselves JSON objects (the standard ss_* fields) are parsed in place.
    """
    try:
        with open(path, "rb") as f:
            size = int.from_bytes(f.read(8), "little", signed=False)
            if size <= 0:
                return {}
            header = json.loads(f.read(size))
    except Exception:
        return {}
    md = header.get("__metadata__") or {}
    if not isinstance(md, dict):
        return {}
    for key, value in list(md.items()):
        if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
            try:
                md[key] = json.loads(value)
            except Exception:
                pass
    return md


def trained_words_from_metadata(md: dict) -> list:
    """ss_tag_frequency ({bucket: {word: count}}) -> [{word, count}], aggregated.

    Buckets come from different training stages ("sks:045", "sks:090", ...);
    per-word counts are summed across buckets.
    """
    freq = md.get("ss_tag_frequency")
    if isinstance(freq, str):
        try:
            freq = json.loads(freq)
        except Exception:
            freq = None
    words = {}
    if isinstance(freq, dict):
        for bucket in freq.values():
            if not isinstance(bucket, dict):
                continue
            for tag, count in bucket.items():
                entry = words.setdefault(tag, {"word": tag, "count": 0})
                try:
                    entry["count"] += int(count)
                except (TypeError, ValueError):
                    pass
    return list(words.values())
