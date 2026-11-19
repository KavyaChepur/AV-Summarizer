"""Export tool node (no LLM): builds the final Markdown document."""
from agents.common import emit, options
from tools.export_tools import build_markdown
from tools.youtube_tools import video_url

NAME = "export"

_SOURCE_LABELS = {"captions": "YouTube captions",
                  "whisper": "Whisper speech-to-text",
                  "gemini": "Gemini speech-to-text"}


def export(state) -> dict:
    emit(NAME, "Building the document…")
    opts = options(state)
    final = state.get("final_outputs") or {
        k: state[k] for k in opts["outputs"] if state.get(k)}

    if state.get("video_id"):
        title = state.get("title") or f"YouTube video {state['video_id']}"
        source_line = f"Source: {video_url(state['video_id'])}"
    else:
        title = state.get("title") or "Audio notes"
        source_line = f"Source file: {state.get('file_name') or 'uploaded file'}"

    note = f"Transcript from {_SOURCE_LABELS.get(state.get('transcript_source'), 'unknown')}"
    if state.get("caption_problem"):
        note += f" (captions were not available: {state['caption_problem']})"

    return {"final_outputs": final,
            "markdown": build_markdown(f"Notes: {title}", source_line, final, note),
            "log": ["Document ready."]}
