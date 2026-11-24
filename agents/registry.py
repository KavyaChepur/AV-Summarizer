"""The analysis (writer) agents, keyed by output id, in display order."""
from agents import (long_summary_agent, mind_map_agent, short_summary_agent,
                    takeaways_agent, timeline_agent)

ANALYSIS_AGENTS = {
    "short_summary": short_summary_agent.AGENT,
    "long_summary": long_summary_agent.AGENT,
    "mind_map": mind_map_agent.AGENT,
    "takeaways": takeaways_agent.AGENT,
    "timeline": timeline_agent.AGENT,
}
