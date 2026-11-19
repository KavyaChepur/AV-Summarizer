"""Shared behaviour for the Summary, Takeaways and Timeline agents.

Each agent can write a first draft and revise its own draft from critic feedback.
They always write in English; the Translator agent handles other languages.
"""
from typing import Callable, List

from langchain_core.prompts import ChatPromptTemplate

from agents.common import emit, options
from llm import client
from tools.text_tools import fmt_time

_SYSTEM = (
    "You are the {role} in a team of agents that turns videos into study notes. "
    "Use ONLY the information in the SOURCE; never invent facts, names, numbers or "
    "timestamps. The SOURCE is a {source_kind} of a {duration} long recording, with "
    "timestamps like [MM:SS]. Write in English, in Markdown. Return only the output "
    "itself, with no preamble."
)

DRAFT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human", "TASK:\n{instruction}\n\nSOURCE:\n{source}"),
])

REVISE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human",
     "TASK:\n{instruction}\n\nA reviewer found these problems in your previous draft:\n"
     "{issues}\n\nRewrite the output so every problem is fixed and everything is "
     "supported by the SOURCE.\n\nPREVIOUS DRAFT:\n{draft}\n\nSOURCE:\n{source}"),
])


class AnalysisAgent:
    def __init__(self, key: str, role: str, instruction: Callable[[dict, dict], str]):
        self.key = key            # state key and output id: summary | takeaways | timeline
        self.role = role          # e.g. "Summary Agent"
        self.instruction = instruction   # (options, state) -> task text

    def _vars(self, state) -> dict:
        return {
            "role": self.role,
            "source_kind": state.get("source_kind", "transcript"),
            "duration": fmt_time(state.get("duration", 0)),
            "instruction": self.instruction(options(state), state),
            "source": state["source_text"],
        }

    def node(self, state) -> dict:
        """LangGraph node: write the first draft."""
        emit(self.key, f"{self.role} is writing…")
        text = client.ask(DRAFT_PROMPT.format_messages(**self._vars(state)))
        return {self.key: text, "log": [f"{self.role} wrote a draft ({len(text.split())} words)."]}

    def revise(self, state, issues: List[str]) -> str:
        emit(self.key, f"{self.role} is revising its draft…")
        return client.ask(REVISE_PROMPT.format_messages(
            **self._vars(state), draft=state[self.key],
            issues="\n".join(f"- {i}" for i in issues)))
