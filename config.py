"""Central settings, read once from the environment / .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_DIR = Path(__file__).resolve().parent


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


# --- LLM -------------------------------------------------------------------
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
# Tried in order when the main model is overloaded (503), out of quota or missing.
GEMINI_FALLBACK_MODELS = [
    m.strip() for m in os.getenv(
        "GEMINI_FALLBACK_MODELS",
        "gemini-2.5-flash,gemini-3.1-flash-lite,gemini-flash-latest,gemini-2.5-flash-lite",
    ).split(",") if m.strip()
]

# Sizes are in characters (~4 characters per token).
SINGLE_PASS_MAX_CHARS = _int("SINGLE_PASS_MAX_CHARS", 60000)
CHUNK_MAX_CHARS = _int("CHUNK_MAX_CHARS", 24000)
REDUCE_MAX_CHARS = _int("REDUCE_MAX_CHARS", 40000)
MAP_CONCURRENCY = _int("MAP_CONCURRENCY", 4)      # parallel chunk calls

# --- Critic ----------------------------------------------------------------
MAX_REVISIONS = _int("MAX_REVISIONS", 2)
TIMESTAMP_TOLERANCE_SECONDS = _int("TIMESTAMP_TOLERANCE_SECONDS", 60)

# --- Speech-to-text --------------------------------------------------------
STT_ENGINE = os.getenv("STT_ENGINE", "whisper")    # "whisper" | "gemini"
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")  # tiny | base | small | medium
AUDIO_PIECE_SECONDS = _int("AUDIO_PIECE_SECONDS", 600)   # Gemini STT piece length
AUDIO_OVERLAP_SECONDS = _int("AUDIO_OVERLAP_SECONDS", 10)

# --- YouTube ---------------------------------------------------------------
CAPTION_LANGUAGES = ("en", "en-US", "en-GB", "hi")

# --- Cache -----------------------------------------------------------------
CACHE_DIR = Path(os.getenv("AVS_CACHE_DIR", PROJECT_DIR / ".cache"))

# --- Chat agent --------------------------------------------------------------
# Transcripts up to this size are given to the chat agent in full (one call per
# question); longer ones are searched locally and only the best passages are sent.
CHAT_FULL_CONTEXT_MAX_CHARS = _int("CHAT_FULL_CONTEXT_MAX_CHARS", 300000)
CHAT_SEARCH_PASSAGES = _int("CHAT_SEARCH_PASSAGES", 8)
CHAT_HISTORY_TURNS = _int("CHAT_HISTORY_TURNS", 6)   # earlier Q&A pairs sent as memory

# --- Mind map ----------------------------------------------------------------
MARKMAP_AUTOLOADER_URL = "https://cdn.jsdelivr.net/npm/markmap-autoloader@0.18.12"

# --- Options shown in the UI ------------------------------------------------
# Display order of the outputs. Each key is produced by the agent of the same name.
OUTPUT_TITLES = {
    "short_summary": "Short summary",
    "long_summary": "Long summary",
    "mind_map": "Mind map",
    "takeaways": "Key takeaways",
    "timeline": "Timestamped notes",
}
DEFAULT_OUTPUTS = ["short_summary", "long_summary", "mind_map", "takeaways"]
LANGUAGES = ["English", "Hindi", "Telugu", "Spanish", "French", "German"]
