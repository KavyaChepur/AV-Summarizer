"""Speech-to-text. Both engines return segments in the same format as captions:
[{"text": str, "start": float, "duration": float}, ...]

Engines:
  whisper - faster-whisper, runs locally (CPU or GPU), free, accurate timestamps.
  gemini  - sends the audio to Gemini in ~10-minute pieces; no local model needed.
"""
import base64
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

import config
from errors import AudioError, LLMError
from llm import client
from tools import audio_tools
from tools.text_tools import parse_time

ProgressFn = Callable[[float, str], None]
WHISPER_SIZES = ["tiny", "base", "small", "medium"]


def _noop(_p: float, _m: str) -> None:
    pass


# ---------------------------------------------------------------- whisper
@lru_cache(maxsize=2)
def _load_whisper(size: str):
    from faster_whisper import WhisperModel
    import ctranslate2

    if ctranslate2.get_cuda_device_count() > 0:
        try:
            return WhisperModel(size, device="cuda", compute_type="float16")
        except Exception:
            pass  # missing CUDA libraries etc. -> fall back to CPU
    return WhisperModel(size, device="cpu", compute_type="int8")


def _whisper(wav: Path, size: str, progress: ProgressFn) -> Tuple[List[Dict], Dict]:
    progress(0.0, f"Loading Whisper '{size}' model…")
    try:
        model = _load_whisper(size)
    except Exception as e:
        raise AudioError(f"Could not load the Whisper '{size}' model: {e}") from e

    segments_iter, info = model.transcribe(str(wav), vad_filter=True, beam_size=5)
    total = info.duration or 1.0
    progress(0.02, f"Detected language: {info.language} "
                   f"({info.language_probability:.0%}). Transcribing…")
    out = []
    for s in segments_iter:  # generator: transcription happens while iterating
        text = s.text.strip()
        if text:
            out.append({"text": text, "start": float(s.start),
                        "duration": max(float(s.end - s.start), 0.0)})
        progress(min(s.end / total, 1.0),
                 f"Transcribed {_mmss(s.end)} of {_mmss(total)}")
    return out, {"language": info.language}


# ---------------------------------------------------------------- gemini
class SpokenSegment(BaseModel):
    start: str = Field(description="Start time from the beginning of THIS clip, as MM:SS")
    end: str = Field(description="End time from the beginning of THIS clip, as MM:SS")
    text: str = Field(description="Exact words spoken, in the original language")


class AudioTranscript(BaseModel):
    language: str = Field(description="Main spoken language, e.g. 'en' or 'hi'")
    segments: List[SpokenSegment]


_GEMINI_STT_PROMPT = (
    "Transcribe this audio clip verbatim in the language that is spoken. Split it into "
    "segments of one or two sentences (at most ~15 seconds each). Give each segment's "
    "start and end time measured from the beginning of this clip, as MM:SS. Do not "
    "summarise, translate or add anything that is not spoken. Skip music and silence."
)


def _gemini_piece(path: Path) -> AudioTranscript:
    data = base64.b64encode(path.read_bytes()).decode()
    msg = HumanMessage(content=[
        {"type": "text", "text": _GEMINI_STT_PROMPT},
        {"type": "audio", "base64": data, "mime_type": "audio/mpeg"},
    ])
    return client.ask_structured(AudioTranscript, [msg])


def _to_segments(t: AudioTranscript) -> List[Dict]:
    out = []
    for s in t.segments:
        try:
            start, end = parse_time(s.start), parse_time(s.end)
        except ValueError:
            continue
        if s.text.strip():
            out.append({"text": s.text.strip(), "start": start,
                        "duration": max(end - start, 0.0)})
    return out


def _gemini(wav: Path, work_dir: Path, progress: ProgressFn) -> Tuple[List[Dict], Dict]:
    progress(0.0, "Cutting audio into pieces for Gemini…")
    pieces = audio_tools.split_audio(wav, work_dir, config.AUDIO_PIECE_SECONDS,
                                     config.AUDIO_OVERLAP_SECONDS)
    per_piece, languages = [], []
    for i, piece in enumerate(pieces):
        progress(i / len(pieces), f"Gemini is transcribing piece {i + 1}/{len(pieces)}…")
        try:
            t = _gemini_piece(piece.path)
        except LLMError as e:
            raise AudioError(f"Gemini could not transcribe piece {i + 1}: {e}") from e
        per_piece.append(_to_segments(t))
        languages.append(t.language)
    progress(1.0, "Merging pieces…")
    language = max(set(languages), key=languages.count) if languages else None
    return audio_tools.merge_piece_segments(pieces, per_piece), {"language": language}


# ---------------------------------------------------------------- public
def transcribe(audio_path: Path, work_dir: Path, engine: str = "whisper",
               whisper_model: str = "base",
               progress: Optional[ProgressFn] = None) -> Tuple[List[Dict], Dict]:
    """Convert audio/video at `audio_path` to segments. Returns (segments, meta)."""
    progress = progress or _noop
    wav = audio_tools.normalize_audio(audio_path, work_dir / "audio_16k.wav")
    if engine == "gemini":
        segments, meta = _gemini(wav, work_dir, progress)
    elif engine == "whisper":
        segments, meta = _whisper(wav, whisper_model, progress)
    else:
        raise AudioError(f"Unknown speech-to-text engine: {engine}")
    if not segments:
        raise AudioError("No speech was found in the audio.")
    return segments, {**meta, "engine": engine}


def _mmss(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"
