"""Chunk-notes agent: map-reduce for long transcripts.

Map: write timestamped notes for every chunk (in parallel).
Reduce: merge groups of notes until everything fits into one request.
"""
from langchain_core.prompts import ChatPromptTemplate

import config
from agents.common import emit
from llm import client
from tools.text_tools import fmt_time

NAME = "chunk_notes"

MAP_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are the Chunk-Notes Agent. You take notes on one part of a long video "
     "transcript. Timestamps look like [MM:SS] or [H:MM:SS]. Use only what is in "
     "the text; never invent facts. Write in English."),
    ("human",
     "This is part {idx} of {total} ({start}–{end}).\n"
     "Extract the key points, facts, examples and conclusions from THIS part only, as "
     "concise bullet points. Start each bullet with the timestamp where that topic "
     "begins, e.g. '- [12:30] ...'.\n\nTRANSCRIPT PART:\n{text}"),
])

REDUCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are the Chunk-Notes Agent. You merge notes from consecutive parts of one "
     "video. Write in English."),
    ("human",
     "Merge these notes into one condensed set: remove repetition, keep chronological "
     "order and keep the [timestamps] exactly as given.\n\nNOTES:\n{notes}"),
])


def _group(parts, limit):
    groups, cur, size = [], [], 0
    for p in parts:
        if cur and size + len(p) > limit:
            groups.append(cur)
            cur, size = [], 0
        cur.append(p)
        size += len(p)
    if cur:
        groups.append(cur)
    return groups


def chunk_notes(state) -> dict:
    chunks = state["chunks"]
    n = len(chunks)
    emit(NAME, f"Taking notes on {n} parts in parallel…")
    prompts = [MAP_PROMPT.format_messages(idx=i, total=n, start=fmt_time(c.start),
                                          end=fmt_time(c.end), text=c.text)
               for i, c in enumerate(chunks, 1)]
    partials = client.ask_many(prompts, temperature=0.2)
    log = [f"Map: wrote notes for {n} parts."]

    level = 0
    notes = "\n\n".join(partials)
    while len(notes) > config.REDUCE_MAX_CHARS and len(partials) > 1:
        groups = _group(partials, config.REDUCE_MAX_CHARS // 2)
        if len(groups) == len(partials):  # can't shrink further
            break
        level += 1
        emit(NAME, f"Condensing notes (pass {level}, {len(groups)} groups)…")
        partials = client.ask_many(
            [REDUCE_PROMPT.format_messages(notes="\n\n".join(g)) for g in groups],
            temperature=0.2)
        notes = "\n\n".join(partials)
        log.append(f"Reduce pass {level}: {len(groups)} groups.")

    return {"source_text": notes, "source_kind": "condensed notes", "log": log}
