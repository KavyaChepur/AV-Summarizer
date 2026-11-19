"""Gemini-backed hierarchical summarization.

To switch to Ollama later, only `call_llm` needs to change.
"""
import os
import time
from typing import Callable, Dict, List, Optional

from google import genai

from youtube_utils import Chunk, chunk_transcript, fmt_time

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
# Characters (~4 chars per token). Tune to your model's context window.
SINGLE_PASS_MAX_CHARS = int(os.getenv("SINGLE_PASS_MAX_CHARS", 60000))
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", 24000))
REDUCE_MAX_CHARS = int(os.getenv("REDUCE_MAX_CHARS", 40000))

LENGTHS = {
    "Short": "Write a concise summary of about 100 words.",
    "Medium": "Write a clear summary of about 250 words, as bullet points.",
    "Detailed": "Write detailed, well-organised notes (about 600-800 words) with "
                "markdown headings per topic and bullet points under each.",
}

ProgressFn = Callable[[float, str], None]


class LLMError(Exception):
    pass


_client = None


def configure(api_key: Optional[str]) -> None:
    global _client
    if not api_key:
        raise LLMError("GOOGLE_API_KEY is not set. Add it to your .env file.")
    _client = genai.Client(api_key=api_key)


def call_llm(prompt: str, retries: int = 3) -> str:
    """Single place that talks to the model, with retry + empty-response handling."""
    if _client is None:
        raise LLMError("Gemini client is not configured.")
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            resp = _client.models.generate_content(model=MODEL_NAME, contents=prompt)
            text = resp.text  # None if blocked / no candidates
            if text and text.strip():
                return text.strip()
            raise LLMError("The model returned no text (it may have been blocked).")
        except Exception as e:  # rate limits, network, etc.
            last = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt * 2)
    raise LLMError(f"Gemini request failed after {retries} attempts: {last}")


# ---------------------------------------------------------------- prompts
def _lang(language: str) -> str:
    return f"Write the output in {language}. " if language and language != "English" else ""


def _chunk_prompt(chunk: Chunk, idx: int, total: int, language: str) -> str:
    return (
        f"You are summarising part {idx} of {total} of a YouTube video transcript "
        f"(timestamps in [MM:SS] or [H:MM:SS]).\n"
        f"Extract the key points, facts, examples and conclusions from THIS part only. "
        f"Write concise bullet points; start each bullet with the timestamp where the "
        f"topic begins, e.g. '- [12:30] ...'. Do not invent information.\n"
        f"{_lang(language)}\n\nTRANSCRIPT PART:\n{chunk.text}"
    )


def _reduce_prompt(notes: str, language: str) -> str:
    return (
        "Below are bullet-point notes from consecutive parts of one video. Merge them into "
        "a single condensed set of notes, removing repetition, keeping chronological order "
        "and keeping the [timestamps].\n"
        f"{_lang(language)}\n\nNOTES:\n{notes}"
    )


def _final_prompt(kind: str, source: str, length: str, language: str) -> str:
    instr = {
        "summary": LENGTHS[length] + " Cover the whole video, not just the beginning.",
        "takeaways": "List the 5-8 most important key takeaways as a numbered list, one "
                     "sentence each.",
        "timeline": "Produce timestamped notes: a chronological list where each line is "
                    "'- **[MM:SS]** topic: one-sentence description'. Use timestamps that "
                    "appear in the source; do not invent them.",
    }[kind]
    return (
        "You are a YouTube video summarizer. Using ONLY the source below, do the "
        f"following.\n{instr}\n{_lang(language)}\n\nSOURCE:\n{source}"
    )


# ---------------------------------------------------------- pipeline
def _hierarchical_notes(chunks: List[Chunk], language: str, progress: ProgressFn) -> str:
    """Map: summarise each chunk. Reduce: merge until the notes fit in one request."""
    n = len(chunks)
    partials = []
    for i, ch in enumerate(chunks, 1):
        progress(0.1 + 0.5 * (i - 1) / n,
                 f"Summarising part {i}/{n} ({fmt_time(ch.start)}–{fmt_time(ch.end)})…")
        partials.append(call_llm(_chunk_prompt(ch, i, n, language)))

    notes = "\n\n".join(partials)
    level = 0
    while len(notes) > REDUCE_MAX_CHARS and len(partials) > 1:
        level += 1
        progress(0.62, f"Condensing notes (pass {level})…")
        groups, cur, size = [], [], 0
        for p in partials:
            if cur and size + len(p) > REDUCE_MAX_CHARS // 2:
                groups.append(cur); cur, size = [], 0
            cur.append(p); size += len(p)
        if cur:
            groups.append(cur)
        if len(groups) == len(partials):  # can't shrink further
            break
        partials = [call_llm(_reduce_prompt("\n\n".join(g), language)) for g in groups]
        notes = "\n\n".join(partials)
    return notes


def summarize_transcript(
    segments: List[Dict],
    length: str = "Medium",
    outputs: tuple = ("summary", "takeaways", "timeline"),
    language: str = "English",
    progress: Optional[ProgressFn] = None,
) -> Dict[str, str]:
    """Return {'summary':…, 'takeaways':…, 'timeline':…} for the requested outputs."""
    progress = progress or (lambda p, m: None)
    progress(0.02, "Preparing transcript…")

    chunks = chunk_transcript(segments, CHUNK_MAX_CHARS)
    total_chars = sum(len(c.text) for c in chunks)

    if total_chars <= SINGLE_PASS_MAX_CHARS:
        source = "\n".join(c.text for c in chunks)   # short video: use transcript directly
    else:
        source = _hierarchical_notes(chunks, language, progress)

    results: Dict[str, str] = {}
    for k, kind in enumerate(outputs):
        progress(0.65 + 0.33 * k / max(len(outputs), 1), f"Generating {kind}…")
        results[kind] = call_llm(_final_prompt(kind, source, length, language))
    progress(1.0, "Done")
    return results
