"""Audio tools: YouTube audio download (yt-dlp), conversion and splitting (ffmpeg)."""
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from errors import AudioError

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".opus", ".flac",
                    ".wma", ".webm", ".mp4", ".mkv", ".mov", ".avi"}


def _require(binary: str) -> str:
    path = shutil.which(binary)
    if not path:
        raise AudioError(f"'{binary}' is not installed or not on PATH. "
                         "Install ffmpeg (e.g. `sudo apt install ffmpeg`).")
    return path


def _run(cmd: List[str], what: str) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-3:]
        raise AudioError(f"{what} failed: {' '.join(tail) or 'unknown ffmpeg error'}")
    return proc


def download_youtube_audio(video_id: str, out_dir: Path) -> Tuple[Path, Dict]:
    """Download the best audio-only stream. Returns (path, {title, duration})."""
    import yt_dlp

    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / f"{video_id}.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}",
                                    download=True)
            path = Path(ydl.prepare_filename(info))
    except Exception as e:
        msg = str(e).replace("ERROR: ", "")
        raise AudioError(f"Could not download the video's audio: {msg}") from e
    if not path.exists():
        raise AudioError("The audio download finished but no file was found.")
    return path, {"title": info.get("title"), "duration": info.get("duration")}


def probe_duration(path: Path) -> float:
    proc = _run([_require("ffprobe"), "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(path)], "Reading audio length")
    try:
        return float(proc.stdout.strip())
    except ValueError as e:
        raise AudioError("Could not read the audio length (is the file valid?).") from e


def normalize_audio(src: Path, dst: Path) -> Path:
    """Convert any audio/video file to 16 kHz mono WAV (what Whisper expects)."""
    _run([_require("ffmpeg"), "-y", "-nostdin", "-i", str(src), "-vn", "-ac", "1",
          "-ar", "16000", "-c:a", "pcm_s16le", str(dst)],
         "Converting the audio (does the file contain an audio track?)")
    return dst


@dataclass
class AudioPiece:
    path: Path
    offset: float      # where this piece starts in the full audio (seconds)
    keep_from: float   # keep segments starting at or after this time…
    keep_to: float     # …and before this time (absolute seconds)


def split_audio(src: Path, out_dir: Path, piece_seconds: int, overlap: int) -> List[AudioPiece]:
    """Cut audio into small mp3 pieces with overlap, for APIs with size limits.

    Neighbouring pieces overlap by `overlap` seconds; each piece "owns" the
    segments that start between the midpoints of its overlaps, so no speech
    is lost or duplicated when the pieces are merged.
    """
    duration = probe_duration(src)
    ffmpeg = _require("ffmpeg")
    pieces: List[AudioPiece] = []
    i = 0
    while i * piece_seconds < duration:
        nominal = i * piece_seconds
        start = max(0.0, nominal - overlap)
        end = min(duration, nominal + piece_seconds)
        out = out_dir / f"piece_{i:03d}.mp3"
        _run([ffmpeg, "-y", "-nostdin", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
              "-i", str(src), "-ac", "1", "-ar", "16000", "-b:a", "32k", str(out)],
             f"Cutting audio piece {i + 1}")
        keep_from = 0.0 if i == 0 else nominal - overlap / 2
        is_last = end >= duration
        keep_to = float("inf") if is_last else nominal + piece_seconds - overlap / 2
        pieces.append(AudioPiece(out, start, keep_from, keep_to))
        i += 1
    return pieces


def merge_piece_segments(pieces: List[AudioPiece], per_piece: List[List[Dict]]) -> List[Dict]:
    """Shift each piece's segments to absolute time and drop overlap duplicates."""
    merged: List[Dict] = []
    for piece, segs in zip(pieces, per_piece):
        for s in segs:
            start = piece.offset + s["start"]
            if piece.keep_from <= start < piece.keep_to:
                merged.append({**s, "start": start})
    merged.sort(key=lambda s: s["start"])
    return merged
