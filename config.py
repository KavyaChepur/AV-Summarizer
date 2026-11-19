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

# --- Options shown in the UI ------------------------------------------------
LENGTHS = {
    "Short": "Write a concise summary of about 100 words.",
    "Medium": "Write a clear summary of about 250 words, as bullet points.",
    "Detailed": "Write detailed, well-organised notes (about 600-800 words) with "
                "markdown headings per topic and bullet points under each.",
}
OUTPUT_TITLES = {
    "summary": "Summary",
    "takeaways": "Key takeaways",
    "timeline": "Timestamped notes",
}
LANGUAGES = ["English", "Hindi", "Telugu", "Spanish", "French", "German"]
