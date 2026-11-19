"""The analysis agents, keyed by output id."""
from agents import summary_agent, takeaways_agent, timeline_agent

ANALYSIS_AGENTS = {
    "summary": summary_agent.AGENT,
    "takeaways": takeaways_agent.AGENT,
    "timeline": timeline_agent.AGENT,
}
