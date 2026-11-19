"""Timeline agent: chronological, timestamped notes."""
from agents.analysis_base import AnalysisAgent
from tools.text_tools import fmt_time


def _instruction(_opts: dict, state: dict) -> str:
    return ("Produce timestamped notes: a chronological list where each line is "
            "'- **[MM:SS]** Topic: one-sentence description' (use [H:MM:SS] after one "
            "hour). Cover the whole recording with one line per topic change. Only use "
            "timestamps that appear in the SOURCE, and none later than "
            f"{fmt_time(state.get('duration', 0))}.")


AGENT = AnalysisAgent("timeline", "Timeline Agent", _instruction)
