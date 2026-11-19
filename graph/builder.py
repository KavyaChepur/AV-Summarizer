"""Builds the LangGraph state graph that connects all agents."""
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from agents.chunk_notes_agent import chunk_notes
from agents.critic_agent import critic, revise
from agents.export_agent import export
from agents.ingestion_agent import ingest
from agents.preprocess import preprocess
from agents.registry import ANALYSIS_AGENTS
from agents.transcription_agent import download_audio, fetch_captions, speech_to_text
from agents.translator_agent import translate
from graph import router
from graph.state import PipelineState

ANALYSIS_NODES = list(ANALYSIS_AGENTS)  # ["summary", "takeaways", "timeline"]

# Labels used by the UI / CLI for each node.
NODE_LABELS = {
    "ingest": "Ingestion agent",
    "fetch_captions": "Transcription agent · captions",
    "download_audio": "Transcription agent · audio download",
    "speech_to_text": "Transcription agent · speech-to-text",
    "preprocess": "Preprocessing",
    "chunk_notes": "Chunk-notes agent",
    "summary": "Summary agent",
    "takeaways": "Takeaways agent",
    "timeline": "Timeline agent",
    "critic": "Critic agent",
    "revise": "Revision (back to writers)",
    "translate": "Translator agent",
    "export": "Export",
    "llm": "Gemini",
}


def build_graph():
    g = StateGraph(PipelineState)

    g.add_node("ingest", ingest)
    g.add_node("fetch_captions", fetch_captions)
    g.add_node("download_audio", download_audio)
    g.add_node("speech_to_text", speech_to_text)
    g.add_node("preprocess", preprocess)
    g.add_node("chunk_notes", chunk_notes)
    for key, agent in ANALYSIS_AGENTS.items():
        g.add_node(key, agent.node)
    g.add_node("critic", critic)
    g.add_node("revise", revise)
    g.add_node("translate", translate)
    g.add_node("export", export)

    g.add_edge(START, "ingest")
    g.add_conditional_edges("ingest", router.after_ingest,
                            ["fetch_captions", "download_audio", "speech_to_text"])
    g.add_conditional_edges("fetch_captions", router.after_captions,
                            ["preprocess", "download_audio"])
    g.add_conditional_edges("download_audio", router.after_download,
                            ["preprocess", "speech_to_text"])
    g.add_edge("speech_to_text", "preprocess")
    g.add_conditional_edges("preprocess", router.after_preprocess,
                            ["chunk_notes", *ANALYSIS_NODES])
    g.add_conditional_edges("chunk_notes", router.select_analysis, ANALYSIS_NODES)
    # The selected analysis agents run in parallel (same step); the critic
    # runs once after all of them have finished.
    for key in ANALYSIS_NODES:
        g.add_edge(key, "critic")
    g.add_conditional_edges("critic", router.after_critic,
                            ["revise", "translate", "export"])
    g.add_edge("revise", "critic")
    g.add_edge("translate", "export")
    g.add_edge("export", END)

    return g.compile()


@lru_cache(maxsize=1)
def get_graph():
    return build_graph()
