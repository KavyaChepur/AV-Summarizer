"""Short Summary agent: the whole recording in a quick read."""
from agents.analysis_base import AnalysisAgent


def _instruction(_opts: dict, _state: dict) -> str:
    return ("Write a SHORT summary of the whole recording in 80-120 words, as one or two "
            "plain paragraphs (no headings, no bullet points). Say what the recording is "
            "about, its main points and its main conclusion.")


AGENT = AnalysisAgent("short_summary", "Short Summary Agent", _instruction)
