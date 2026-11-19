"""Ingestion agent: checks the input and works out what kind of source it is."""
from pathlib import Path

from agents.common import emit
from errors import InvalidInput
from tools import cache
from tools.audio_tools import AUDIO_EXTENSIONS
from tools.youtube_tools import extract_video_id

NAME = "ingestion"


def ingest(state) -> dict:
    if state.get("source_type") == "youtube":
        video_id = extract_video_id(state.get("url") or "")
        emit(NAME, f"YouTube video {video_id}")
        return {"video_id": video_id, "source_key": f"yt_{video_id}",
                "log": [f"Recognised YouTube video `{video_id}`."]}

    if state.get("source_type") == "upload":
        path = Path(state.get("file_path") or "")
        name = state.get("file_name") or path.name
        if not path.is_file():
            raise InvalidInput("The uploaded file could not be found.")
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            raise InvalidInput(f"Unsupported file type '{path.suffix}'. Supported: "
                               + ", ".join(sorted(AUDIO_EXTENSIONS)))
        size = path.stat().st_size
        if size == 0:
            raise InvalidInput("The uploaded file is empty.")
        emit(NAME, f"Checking {name}")
        return {"source_key": f"file_{cache.file_fingerprint(path)}",
                "title": Path(name).stem,
                "log": [f"Received file `{name}` ({size / 1e6:.1f} MB)."]}

    raise InvalidInput("Give either a YouTube link or an audio/video file.")
