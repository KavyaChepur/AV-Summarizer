"""Helpers shared by all agents."""
from typing import Optional

from langgraph.config import get_stream_writer

import config


def emit(agent: str, message: str, progress: Optional[float] = None) -> None:
    """Send a live progress event to whoever is streaming the graph (the UI/CLI)."""
    try:
        writer = get_stream_writer()
    except Exception:  # called outside a running graph (e.g. in tests)
        return
    writer({"agent": agent, "message": message, "progress": progress})


def options(state) -> dict:
    """Options with defaults filled in."""
    defaults = {
        "length": "Medium",
        "outputs": ["summary", "takeaways", "timeline"],
        "language": "English",
        "stt_engine": config.STT_ENGINE,
        "whisper_model": config.WHISPER_MODEL,
        "force_stt": False,
        "allow_stt_fallback": True,
        "use_critic": True,
    }
    return {**defaults, **(state.get("options") or {})}
