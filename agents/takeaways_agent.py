"""Takeaways agent: the most important points as a numbered list."""
from agents.analysis_base import AnalysisAgent


def _instruction(_opts: dict, _state: dict) -> str:
    return ("List the 5-8 most important key takeaways as a numbered list, one clear "
            "sentence each. Prefer insights, conclusions and actionable advice over "
            "minor details.")


AGENT = AnalysisAgent("takeaways", "Takeaways Agent", _instruction)
