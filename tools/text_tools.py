"""Transcript cleaning, timestamp helpers and timestamp-aware chunking.

A "segment" everywhere in this project is {"text": str, "start": float,
"duration": float}, the format returned by both captions and speech-to-text.
"""
import html
import re
from dataclasses import dataclass
from typing import Dict, List

_NOISE_RE = re.compile(r"\[(?:music|applause|laughter|inaudible|__)[^\]]*\]", re.I)
_TS_RE = re.compile(r"\[(?:\*\*)?(\d{1,2}(?::\d{2}){1,2})(?:\*\*)?\]")


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = _NOISE_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def fmt_time(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def parse_time(stamp: str) -> float:
    """'12:30' -> 750.0, '1:02:03' -> 3723.0. Raises ValueError on bad input."""
    parts = [int(p) for p in stamp.strip().split(":")]
    if not 2 <= len(parts) <= 3:
        raise ValueError(f"bad timestamp: {stamp!r}")
    seconds = 0
    for p in parts:
        seconds = seconds * 60 + p
    return float(seconds)


def find_timestamps(text: str) -> List[float]:
    """All [MM:SS] / [H:MM:SS] (optionally **bold**) timestamps in a text, in order."""
    out = []
    for m in _TS_RE.finditer(text):
        try:
            out.append(parse_time(m.group(1)))
        except ValueError:
            pass
    return out


def transcript_duration(segments: List[Dict]) -> float:
    if not segments:
        return 0.0
    return max(s["start"] + s["duration"] for s in segments)


@dataclass
class Chunk:
    text: str       # lines like "[MM:SS] words..."
    start: float
    end: float


def build_timestamped_lines(segments: List[Dict], line_seconds: float = 30.0) -> List[Dict]:
    """Merge tiny caption fragments into ~30 s lines, each tagged with a start time."""
    lines, buf, line_start = [], [], None
    for seg in segments:
        txt = clean_text(seg["text"])
        if not txt:
            continue
        if line_start is None:
            line_start = seg["start"]
        buf.append(txt)
        if seg["start"] + seg["duration"] - line_start >= line_seconds:
            lines.append({"start": line_start, "text": " ".join(buf)})
            buf, line_start = [], None
    if buf:
        lines.append({"start": line_start, "text": " ".join(buf)})
    return lines


def render_transcript(segments: List[Dict]) -> str:
    """Readable transcript: one '[MM:SS] text' line per ~30 s."""
    return "\n".join(f"[{fmt_time(ln['start'])}] {ln['text']}"
                     for ln in build_timestamped_lines(segments))


def chunk_transcript(segments: List[Dict], max_chars: int) -> List[Chunk]:
    """Split on line boundaries so no chunk exceeds max_chars (approximately)."""
    lines = build_timestamped_lines(segments)
    chunks: List[Chunk] = []
    cur, cur_len, cur_start = [], 0, None
    for ln in lines:
        rendered = f"[{fmt_time(ln['start'])}] {ln['text']}"
        if cur and cur_len + len(rendered) > max_chars:
            chunks.append(Chunk("\n".join(cur), cur_start, ln["start"]))
            cur, cur_len, cur_start = [], 0, None
        if cur_start is None:
            cur_start = ln["start"]
        cur.append(rendered)
        cur_len += len(rendered) + 1
    if cur:
        chunks.append(Chunk("\n".join(cur), cur_start, transcript_duration(segments)))
    return chunks
