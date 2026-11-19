"""Critic / QA agent: checks every output against the source.

Two kinds of checks:
  1. LLM review - unsupported claims, missing major topics, wrong format.
  2. Code check - timeline timestamps must exist in the source, be in order and
     not exceed the recording's length.

Failed outputs are sent back to the agent that wrote them (`revise` node), at
most MAX_REVISIONS times.
"""
from bisect import bisect_left
from concurrent.futures import ThreadPoolExecutor
from typing import List, Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

import config
from agents.common import emit, options
from agents.registry import ANALYSIS_AGENTS
from errors import LLMError
from llm import client
from tools.text_tools import find_timestamps, fmt_time

NAME = "critic"


class OutputReview(BaseModel):
    output: Literal["summary", "takeaways", "timeline"]
    passed: bool = Field(description="true if there are no significant problems")
    issues: List[str] = Field(default_factory=list,
                              description="Specific, actionable problems; empty if passed")


class Critique(BaseModel):
    reviews: List[OutputReview]


CRITIC_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are the Critic Agent, a strict fact-checker for video notes written by other "
     "agents. Compare each OUTPUT with the SOURCE (a {source_kind} of a {duration} long "
     "recording). Report only significant problems:\n"
     "- claims, names or numbers that are not supported by the SOURCE\n"
     "- timestamps that do not appear in the SOURCE\n"
     "- major topics of the recording that are missing (the summary must cover the whole "
     "recording)\n"
     "- the output ignores its TASK (wrong format or far off the requested length)\n"
     "Ignore minor wording and style. Each issue must say exactly what is wrong and how "
     "to fix it. Review every output listed."),
    ("human", "{outputs}\n\nSOURCE:\n{source}"),
])


def validate_timestamps(text: str, source_text: str, duration: float,
                        tolerance: float = config.TIMESTAMP_TOLERANCE_SECONDS) -> List[str]:
    """Code-only checks for a timestamped output. Returns a list of issues."""
    stamps = find_timestamps(text)
    if not stamps:
        return ["No [MM:SS] timestamps were found; every line needs one."]
    issues = []
    too_late = [t for t in stamps if t > duration + 5]
    if too_late:
        issues.append("These timestamps are later than the end of the recording "
                      f"({fmt_time(duration)}): " + ", ".join(fmt_time(t) for t in too_late))
    source = sorted(find_timestamps(source_text))
    if source:
        def distance(t):
            i = bisect_left(source, t)
            return min(abs(t - source[j]) for j in (i - 1, i) if 0 <= j < len(source))
        unsupported = [t for t in stamps if t <= duration + 5 and distance(t) > tolerance]
        if unsupported:
            issues.append("These timestamps don't match any point in the source; use the "
                          "source's timestamps: " + ", ".join(fmt_time(t) for t in unsupported))
    if any(b < a - 1 for a, b in zip(stamps, stamps[1:])):
        issues.append("The timestamps are not in chronological order.")
    return issues


def critic(state) -> dict:
    opts = options(state)
    keys = [k for k in opts["outputs"] if state.get(k)]
    if not opts["use_critic"]:
        return {"reviews": {k: {"passed": True, "issues": []} for k in keys},
                "log": ["Critic skipped (turned off in options)."]}

    previous = state.get("reviews") or {}
    to_review = [k for k in keys if not previous.get(k, {}).get("passed")]
    emit(NAME, f"Checking {', '.join(to_review)} against the source…")

    code_issues = {k: [] for k in to_review}
    if "timeline" in code_issues:
        code_issues["timeline"] = validate_timestamps(
            state["timeline"], state["source_text"], state.get("duration", 0))

    log = []
    llm_reviews = {}
    outputs = "\n\n".join(
        f"### OUTPUT: {k}\nTASK: {ANALYSIS_AGENTS[k].instruction(opts, state)}\n\n{state[k]}"
        for k in to_review)
    try:
        critique = client.ask_structured(Critique, CRITIC_PROMPT.format_messages(
            source_kind=state.get("source_kind", "transcript"),
            duration=fmt_time(state.get("duration", 0)),
            outputs=outputs, source=state["source_text"]))
        llm_reviews = {r.output: r for r in critique.reviews}
    except LLMError as e:
        log.append(f"⚠️ LLM review failed ({e}); only the code checks were applied.")

    reviews = dict(previous)
    for k in to_review:
        r = llm_reviews.get(k)
        issues = (list(r.issues) if r is not None and not r.passed else []) + code_issues[k]
        reviews[k] = {"passed": not issues, "issues": issues}

    failed = [k for k in to_review if not reviews[k]["passed"]]
    log.append("Critic: all outputs passed." if not failed else
               "Critic: issues found in " + ", ".join(
                   f"{k} ({len(reviews[k]['issues'])})" for k in failed) + ".")
    return {"reviews": reviews, "log": log}


def revise(state) -> dict:
    """Node: send each failed output back to the agent that wrote it."""
    reviews = state["reviews"]
    failed = [k for k, r in reviews.items() if not r["passed"]]
    with ThreadPoolExecutor(max_workers=len(failed) or 1) as pool:
        futures = {k: pool.submit(ANALYSIS_AGENTS[k].revise, state, reviews[k]["issues"])
                   for k in failed}
        updates = {k: f.result() for k, f in futures.items()}
    count = state.get("revision_count", 0) + 1
    return {**updates, "revision_count": count,
            "log": [f"Revision round {count}: rewrote " + ", ".join(failed) + "."]}
