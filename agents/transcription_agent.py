"""Transcription agent: gets a timestamped transcript.

YouTube: try captions first; if they're missing, download the audio and run
speech-to-text. Uploaded files: always speech-to-text. Transcripts produced by
speech-to-text are cached on disk so the same audio is never transcribed twice.
"""
import shutil
import tempfile
from pathlib import Path

import config
from agents.common import emit, options
from errors import TranscriptError
from tools import cache
from tools.audio_tools import download_youtube_audio
from tools.stt_tools import transcribe
from tools.youtube_tools import fetch_transcript

NAME = "transcription"


def _stt_cache_key(state) -> str:
    opts = options(state)
    engine = opts["stt_engine"]
    suffix = f"_{opts['whisper_model']}" if engine == "whisper" else ""
    return f"{state['source_key']}_{engine}{suffix}"


def _from_cache(state):
    hit = cache.get(_stt_cache_key(state))
    if not hit:
        return None
    return {"segments": hit["segments"], "transcript_source": hit["engine"],
            "detected_language": hit.get("language"),
            "title": state.get("title") or hit.get("title"),
            "log": [f"Reused the cached {hit['engine']} transcript "
                    f"({len(hit['segments'])} segments)."]}


def fetch_captions(state) -> dict:
    """Node: YouTube captions."""
    emit(NAME, "Fetching YouTube captions…")
    try:
        segments = fetch_transcript(state["video_id"], config.CAPTION_LANGUAGES)
    except TranscriptError as e:
        if not e.stt_can_help:
            raise
        if not options(state)["allow_stt_fallback"]:
            raise TranscriptError(f"{e} Turn on the speech-to-text fallback to "
                                  "transcribe the audio instead.", e.reason) from e
        return {"caption_problem": str(e),
                "log": [f"Captions unavailable ({e}) → switching to speech-to-text."]}
    return {"segments": segments, "transcript_source": "captions",
            "log": [f"Fetched captions: {len(segments)} segments."]}


def download_audio(state) -> dict:
    """Node: download a YouTube video's audio (skipped if a cached transcript exists)."""
    cached = _from_cache(state)
    if cached:
        return cached
    emit(NAME, "Downloading the audio with yt-dlp…")
    work_dir = tempfile.mkdtemp(prefix="avs_")
    try:
        path, info = download_youtube_audio(state["video_id"], Path(work_dir))
    except Exception:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise
    size = path.stat().st_size / 1e6
    return {"work_dir": work_dir, "audio_path": str(path),
            "title": info.get("title") or state.get("title"),
            "log": [f"Downloaded audio ({size:.1f} MB)."]}


def speech_to_text(state) -> dict:
    """Node: convert audio to a timestamped transcript."""
    cached = _from_cache(state)
    if cached:
        return cached

    opts = options(state)
    engine, model = opts["stt_engine"], opts["whisper_model"]
    audio = Path(state.get("audio_path") or state["file_path"])
    work_dir = Path(state.get("work_dir") or tempfile.mkdtemp(prefix="avs_"))
    label = f"Whisper ({model})" if engine == "whisper" else "Gemini"
    emit(NAME, f"Transcribing with {label}…", 0.0)
    try:
        segments, meta = transcribe(audio, work_dir, engine=engine, whisper_model=model,
                                    progress=lambda p, m: emit(NAME, m, p))
    finally:
        # Only our own temp folder is removed. An uploaded file lives elsewhere.
        shutil.rmtree(work_dir, ignore_errors=True)

    cache.put(_stt_cache_key(state), {"segments": segments, "engine": engine,
                                      "language": meta.get("language"),
                                      "title": state.get("title")})
    return {"segments": segments, "transcript_source": engine,
            "detected_language": meta.get("language"),
            "work_dir": None, "audio_path": None,
            "log": [f"Transcribed with {label}: {len(segments)} segments, "
                    f"language '{meta.get('language')}'."]}
