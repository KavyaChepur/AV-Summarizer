"""Orchestrator: the routing decisions between agents.

Each function looks at the shared state and returns the next node(s).
Returning a list of nodes runs them in parallel.
"""
from typing import List, Union

import config
from agents.common import options

Next = Union[str, List[str]]


def after_ingest(state) -> str:
    if state.get("source_type") == "upload":
        return "speech_to_text"
    if options(state)["force_stt"]:
        return "download_audio"
    return "fetch_captions"


def after_captions(state) -> str:
    return "preprocess" if state.get("segments") else "download_audio"


def after_download(state) -> str:
    # download_audio fills `segments` directly when a cached transcript exists
    return "preprocess" if state.get("segments") else "speech_to_text"


def select_analysis(state) -> List[str]:
    """Fan out to the analysis agents the user asked for."""
    return [k for k in options(state)["outputs"]
            if k in ("summary", "takeaways", "timeline")]


def after_preprocess(state) -> Next:
    return "chunk_notes" if state.get("needs_map_reduce") else select_analysis(state)


def after_critic(state) -> str:
    reviews = state.get("reviews") or {}
    failed = [k for k, r in reviews.items() if not r["passed"]]
    if failed and state.get("revision_count", 0) < config.MAX_REVISIONS:
        return "revise"
    return "translate" if options(state)["language"] != "English" else "export"
