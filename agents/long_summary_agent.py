"""Long Summary agent: a detailed, section-by-section summary."""
from agents.analysis_base import AnalysisAgent


def _instruction(_opts: dict, _state: dict) -> str:
    return ("Write a LONG, detailed summary of the whole recording (about 600-900 words; "
            "shorter only if the recording itself is very short). Start with one "
            "overview paragraph. Then add a '###' Markdown heading for each major topic, "
            "in the order they are discussed, with short paragraphs or bullet points "
            "under each covering the explanations, examples, numbers and conclusions. "
            "Cover the recording from beginning to end.")


AGENT = AnalysisAgent("long_summary", "Long Summary Agent", _instruction)
