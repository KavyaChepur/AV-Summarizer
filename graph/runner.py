"""Runs the graph and turns LangGraph's stream into simple events for a UI.

Events:
  {"type": "progress", "agent": str, "message": str, "progress": float | None}
  {"type": "step", "node": str, "log": [str]}      # a node finished
  {"type": "done", "state": dict}                  # final state
"""
from typing import Dict, Iterator

from graph.builder import get_graph


def stream_pipeline(initial_state: Dict) -> Iterator[Dict]:
    final = dict(initial_state)
    for mode, data in get_graph().stream(initial_state,
                                         stream_mode=["custom", "updates", "values"]):
        if mode == "custom":
            yield {"type": "progress", **data}
        elif mode == "updates":
            for node, update in data.items():
                yield {"type": "step", "node": node, "log": (update or {}).get("log", [])}
        else:
            final = data
    yield {"type": "done", "state": final}
