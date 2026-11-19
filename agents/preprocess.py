"""Preprocessing tool node (no LLM): clean, add timestamps, split into chunks."""
import config
from agents.common import emit
from errors import TranscriptError
from tools.text_tools import chunk_transcript, fmt_time, render_transcript, transcript_duration

NAME = "preprocess"


def preprocess(state) -> dict:
    emit(NAME, "Cleaning and chunking the transcript…")
    segments = state.get("segments") or []
    chunks = chunk_transcript(segments, config.CHUNK_MAX_CHARS)
    if not chunks:
        raise TranscriptError("The transcript has no usable text.", "empty")

    duration = transcript_duration(segments)
    total = sum(len(c.text) for c in chunks)
    needs_map_reduce = total > config.SINGLE_PASS_MAX_CHARS
    update = {
        "duration": duration,
        "transcript_text": render_transcript(segments),
        "chunks": chunks,
        "needs_map_reduce": needs_map_reduce,
        "log": [f"Transcript is ~{fmt_time(duration)} long, {total:,} characters, "
                f"{len(chunks)} chunk(s) → "
                + ("map-reduce needed." if needs_map_reduce else "single pass.")],
    }
    if not needs_map_reduce:
        update["source_text"] = "\n".join(c.text for c in chunks)
        update["source_kind"] = "transcript"
    return update
