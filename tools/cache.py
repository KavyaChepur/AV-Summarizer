"""Tiny JSON disk cache, so the same audio is never transcribed twice."""
import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import config


def file_fingerprint(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def _path(key: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
    return config.CACHE_DIR / "transcripts" / f"{safe}.json"


def get(key: str) -> Optional[Any]:
    p = _path(key)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def put(key: str, value: Any) -> None:
    p = _path(key)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass  # caching is best-effort
