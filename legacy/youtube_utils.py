"""YouTube helpers: URL parsing, transcript fetching, cleaning, chunking."""
import html
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
from urllib.parse import parse_qs, urlparse

import youtube_transcript_api as yta
from youtube_transcript_api import YouTubeTranscriptApi

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_YT_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com",
             "music.youtube.com", "youtu.be", "www.youtu.be",
             "youtube-nocookie.com", "www.youtube-nocookie.com"}
_PATH_PREFIXES = ("shorts", "embed", "live", "v", "e")


class InvalidYouTubeURL(ValueError):
    pass


class TranscriptError(Exception):
    """Raised with a user-friendly message when a transcript can't be obtained."""


# --------------------------------------------------------------------------
# 1. URL validation
# --------------------------------------------------------------------------
def extract_video_id(url: str) -> str:
    """Return the 11-char video ID from watch / youtu.be / shorts / embed / live
    URLs (or a bare video ID). Raises InvalidYouTubeURL otherwise."""
    if not url or not url.strip():
        raise InvalidYouTubeURL("Please enter a YouTube link.")
    url = url.strip()

    if _ID_RE.match(url):
        return url

    if "://" not in url:
        url = "https://" + url
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host not in _YT_HOSTS:
        raise InvalidYouTubeURL("That doesn't look like a YouTube link.")

    candidate: Optional[str] = None
    if host.endswith("youtu.be"):
        candidate = parsed.path.strip("/").split("/")[0]
    else:
        parts = [p for p in parsed.path.split("/") if p]
        if parts and parts[0] == "watch":
            candidate = parse_qs(parsed.query).get("v", [None])[0]
        elif len(parts) >= 2 and parts[0] in _PATH_PREFIXES:
            candidate = parts[1]

    if not candidate or not _ID_RE.match(candidate):
        raise InvalidYouTubeURL("Couldn't find a valid video ID in that link.")
    return candidate


# --------------------------------------------------------------------------
# 2. Transcript fetching with error handling
# --------------------------------------------------------------------------
def _normalise(snippets) -> List[Dict]:
    """Convert library output (old: dicts, new: objects) to plain dicts."""
    out = []
    for s in snippets:
        if isinstance(s, dict):
            text, start, dur = s.get("text", ""), s.get("start", 0), s.get("duration", 0)
        else:
            text, start, dur = s.text, s.start, s.duration
        out.append({"text": text, "start": float(start), "duration": float(dur)})
    return out


def _fetch_raw(video_id: str, languages: Sequence[str]):
    """Fetch transcript, supporting both youtube_transcript_api 0.x and 1.x."""
    if hasattr(YouTubeTranscriptApi, "get_transcript"):  # 0.x
        try:
            return YouTubeTranscriptApi.get_transcript(video_id, languages=list(languages))
        except yta.NoTranscriptFound:
            transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
            return next(iter(transcripts)).fetch()
    api = YouTubeTranscriptApi()  # 1.x
    try:
        return api.fetch(video_id, languages=list(languages))
    except yta.NoTranscriptFound:
        # fall back to whatever language exists
        return next(iter(api.list(video_id))).fetch()


def _exc(*names):
    """Tuple of exception classes that exist in the installed library version."""
    return tuple(getattr(yta, n) for n in names if hasattr(yta, n))


def fetch_transcript(video_id: str, languages: Sequence[str] = ("en",)) -> List[Dict]:
    """Return a list of {text,start,duration}. Raises TranscriptError."""
    try:
        segments = _normalise(_fetch_raw(video_id, languages))
    except _exc("TranscriptsDisabled") as e:
        raise TranscriptError("Captions are disabled for this video.") from e
    except _exc("VideoUnavailable", "VideoUnplayable") as e:
        raise TranscriptError("This video is unavailable (private, deleted or region-locked).") from e
    except _exc("AgeRestricted") as e:
        raise TranscriptError("This video is age-restricted, so its transcript can't be fetched.") from e
    except _exc("InvalidVideoId") as e:
        raise TranscriptError("Invalid video ID.") from e
    except _exc("RequestBlocked", "IpBlocked") as e:
        raise TranscriptError(
            "YouTube blocked the transcript request (common on cloud/VPN IPs). "
            "Try again later or from a different network.") from e
    except (StopIteration,) + _exc("NoTranscriptFound") as e:
        raise TranscriptError("No transcript is available for this video.") from e
    except Exception as e:
        raise TranscriptError(f"Could not fetch the transcript: {e}") from e

    if not segments:
        raise TranscriptError("The transcript for this video is empty.")
    return segments


# --------------------------------------------------------------------------
# 3. Preprocessing
# --------------------------------------------------------------------------
_NOISE_RE = re.compile(r"\[(?:music|applause|laughter|inaudible|__)[^\]]*\]", re.I)


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = _NOISE_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def fmt_time(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# --------------------------------------------------------------------------
# 4. Chunking (timestamp-aware)
# --------------------------------------------------------------------------
@dataclass
class Chunk:
    text: str       # lines like "[MM:SS] words..."
    start: float
    end: float


def build_timestamped_lines(segments: List[Dict], line_seconds: float = 30.0) -> List[Dict]:
    """Merge tiny caption fragments into ~30s lines, each tagged with a start time."""
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


def chunk_transcript(segments: List[Dict], max_chars: int) -> List[Chunk]:
    """Split on line boundaries so no chunk exceeds max_chars (approx)."""
    lines = build_timestamped_lines(segments)
    chunks: List[Chunk] = []
    cur, cur_len, cur_start = [], 0, None
    for i, ln in enumerate(lines):
        rendered = f"[{fmt_time(ln['start'])}] {ln['text']}"
        if cur and cur_len + len(rendered) > max_chars:
            chunks.append(Chunk("\n".join(cur), cur_start, ln["start"]))
            cur, cur_len, cur_start = [], 0, None
        if cur_start is None:
            cur_start = ln["start"]
        cur.append(rendered)
        cur_len += len(rendered) + 1
    if cur:
        last = segments[-1]
        chunks.append(Chunk("\n".join(cur), cur_start, last["start"] + last["duration"]))
    return chunks
