"""YouTube helpers: link parsing and caption download."""
import re
from typing import Dict, List, Optional, Sequence
from urllib.parse import parse_qs, urlparse

import youtube_transcript_api as yta
from youtube_transcript_api import YouTubeTranscriptApi

from errors import InvalidYouTubeURL, TranscriptError

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_YT_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com",
             "music.youtube.com", "youtu.be", "www.youtu.be",
             "youtube-nocookie.com", "www.youtube-nocookie.com"}
_PATH_PREFIXES = ("shorts", "embed", "live", "v", "e")


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


def video_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


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
    """Fetch captions, supporting both youtube_transcript_api 0.x and 1.x."""
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
        return next(iter(api.list(video_id))).fetch()  # any available language


def _exc(*names):
    """Tuple of exception classes that exist in the installed library version."""
    return tuple(getattr(yta, n) for n in names if hasattr(yta, n))


def fetch_transcript(video_id: str, languages: Sequence[str] = ("en",)) -> List[Dict]:
    """Return a list of segments. Raises TranscriptError (with a `reason`)."""
    try:
        segments = _normalise(_fetch_raw(video_id, languages))
    except _exc("TranscriptsDisabled") as e:
        raise TranscriptError("Captions are disabled for this video.", "disabled") from e
    except _exc("VideoUnavailable", "VideoUnplayable") as e:
        raise TranscriptError("This video is unavailable (private, deleted or "
                              "region-locked).", "unavailable") from e
    except _exc("AgeRestricted") as e:
        raise TranscriptError("This video is age-restricted, so it can't be "
                              "fetched.", "age_restricted") from e
    except _exc("InvalidVideoId") as e:
        raise TranscriptError("Invalid video ID.", "invalid") from e
    except _exc("RequestBlocked", "IpBlocked") as e:
        raise TranscriptError("YouTube blocked the caption request (common on "
                              "cloud/VPN IPs).", "blocked") from e
    except (StopIteration,) + _exc("NoTranscriptFound") as e:
        raise TranscriptError("No captions are available for this video.", "not_found") from e
    except Exception as e:
        raise TranscriptError(f"Could not fetch the captions: {e}", "other") from e

    if not any(s["text"].strip() for s in segments):
        raise TranscriptError("The captions for this video are empty.", "empty")
    return segments
