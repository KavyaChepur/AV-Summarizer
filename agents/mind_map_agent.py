"""Mind Map agent: the recording's ideas as a hierarchy.

It writes a nested Markdown outline; tools/mindmap_tools.py turns that into the
interactive mind map, the Mermaid diagram and the PDF drawing.
"""
from agents.analysis_base import AnalysisAgent


def _instruction(_opts: dict, _state: dict) -> str:
    return (
        "Create a MIND MAP of the recording as a nested Markdown bullet list, indented "
        "with 2 spaces per level:\n"
        "- The first line is the central topic (at most 6 words), not indented.\n"
        "- Under it, 3-7 main branches for the main themes, in the order they come up.\n"
        "- Under each branch, 2-5 sub-points; add a 4th level only for important details.\n"
        "- Every node is a short phrase (at most 8 words), not a sentence.\n"
        "- No timestamps, bold, numbering or text outside the list.\n"
        "Example of the format:\n"
        "- Central topic\n"
        "  - Theme one\n"
        "    - Key point\n"
        "    - Key point\n"
        "  - Theme two\n"
        "    - Key point")


AGENT = AnalysisAgent("mind_map", "Mind Map Agent", _instruction)
