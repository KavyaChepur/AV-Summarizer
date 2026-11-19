"""Summary agent: a Short / Medium / Detailed summary of the whole recording."""
from agents.analysis_base import AnalysisAgent
from config import LENGTHS


def _instruction(opts: dict, _state: dict) -> str:
    return (LENGTHS.get(opts["length"], LENGTHS["Medium"])
            + " Cover the whole recording from beginning to end, not just the start.")


AGENT = AnalysisAgent("summary", "Summary Agent", _instruction)
