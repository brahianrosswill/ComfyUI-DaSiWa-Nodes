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


def merge_civitai(info: dict, civ) -> bool:
    """Merge a Civitai model-versions API response into `info`. Returns True if changed.

    `civ` may be None (model not found on Civitai) — then nothing is touched.
    """
    if not civ:
        return False
    changed = False
    model = civ.get("model") if isinstance(civ.get("model"), dict) else {}
    if "name" not in info and civ.get("name"):
        model_name = model.get("name", "")
        version_name = civ["name"]
        info["name"] = f"{model_name} - {version_name}" if model_name else version_name
        changed = True
    for key in ("type", "baseModel"):
        value = civ.get(key) or model.get(key)
        if key not in info and value:
            info[key] = value
            changed = True
    word_map = {w["word"]: w for w in info.get("trainedWords", []) if isinstance(w, dict)}
    merged = False
    for word in list(civ.get("triggerWords", [])) + list(civ.get("trainedWords", [])):
        if not isinstance(word, str) or not word:
            continue
        entry = word_map.setdefault(word, {"word": word})
        entry["civitai"] = True
        merged = True
    if merged:
        for w in word_map.values():
            w.setdefault("count", 0)
        info["trainedWords"] = sorted(word_map.values(), key=lambda w: -w["count"])
        changed = True
    if civ.get("modelId") or civ.get("id"):
        link = f"https://civitai.com/models/{civ.get('modelId', '')}"
        if civ.get("id"):
            link += f"?modelVersionId={civ['id']}"
        info["links"] = info.get("links", []) + [link]
        changed = True
    if civ.get("images"):
        existing_urls = {im.get("url") for im in info.get("images", []) if isinstance(im, dict)}
        for img in civ["images"]:
            if not isinstance(img, dict):
                continue
            url = img.get("url")
            if not url or url in existing_urls:
                continue
            meta = img.get("meta") or {}
            info.setdefault("images", []).append({
                "url": url,
                "type": img.get("type"),
                "width": img.get("width"),
                "height": img.get("height"),
                "seed": meta.get("seed"),
                "positive": meta.get("prompt"),
                "negative": meta.get("negativePrompt"),
                "steps": meta.get("steps"),
                "sampler": meta.get("sampler"),
                "cfg": meta.get("cfgScale"),
                "model": meta.get("Model"),
            })
            changed = True
    return changed


def cache_read(sha: str):
    """Cached info for a file sha256, or None."""
    path = os.path.join(CACHE_DIR, f"{sha}.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def cache_write(sha: str, data: dict):
    """Persist info next to the nodepack. Never raises (cache is convenience)."""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(os.path.join(CACHE_DIR, f"{sha}.json"), "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass
