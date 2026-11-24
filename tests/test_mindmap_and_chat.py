"""Mind map tools, transcript search and the chat agent (fake LLM, no API calls)."""
import pytest

import config
from agents import chat_agent
from errors import InvalidInput
from llm import client
from tools import mindmap_tools as mm
from tools.retrieval import BM25Index, make_passages, tokenize
from tools.text_tools import link_timestamps

OUTLINE = """Here is the mind map:
- Machine Learning
  - Supervised learning
    - **Labelled data** [02:10]
    - Regression (numbers)
  - Unsupervised learning
    - Clustering
"""


# ---------------------------------------------------------------- mind map
def test_parse_outline_cleans_labels_and_ignores_prose():
    tree = mm.to_tree(OUTLINE)
    assert tree.label == "Machine Learning"
    assert [c.label for c in tree.children] == ["Supervised learning", "Unsupervised learning"]
    assert [c.label for c in tree.children[0].children] == ["Labelled data",
                                                           "Regression (numbers)"]
    assert tree.depth() == 3 and tree.leaves() == 3
    assert mm.validate_outline(OUTLINE) == []


def test_parse_handles_headings_and_four_space_indent():
    text = "# Root\n## Branch A\n    - a1\n    - a2\n## Branch B\n- b1"
    tree = mm.to_tree(text)
    assert tree.label == "Root"
    assert [c.label for c in tree.children] == ["Branch A", "Branch B"]
    assert [c.label for c in tree.children[1].children] == ["b1"]


def test_several_top_level_items_get_a_root():
    text = "- A\n  - a\n- B\n  - b"
    assert mm.to_tree(text, fallback_root="Talk").label == "Talk"
    assert any("top-level items" in i for i in mm.validate_outline(text))


@pytest.mark.parametrize("text, problem", [
    ("Just a paragraph of prose.", "nested Markdown bullet list"),
    ("- Root\n  - Only branch", "at least 2 main branches"),
    ("- Root\n  - A\n  - " + "word " * 30, "short phrase"),
    ("- R\n  - A\n    - B\n      - C\n        - D\n          - E\n  - F", "too deep"),
])
def test_validate_outline_problems(text, problem):
    assert any(problem in i for i in mm.validate_outline(text))


def test_mermaid_and_html_are_safe():
    tree = mm.to_tree("- Root (main)\n  - A [x] \"q\"\n  - B </script>")
    mermaid = mm.to_mermaid(tree)
    assert mermaid.startswith("mindmap\n  root((Root  main))")
    assert "[" not in mermaid.split("\n", 2)[2] and '"' not in mermaid
    page = mm.to_html(tree, "T <b>")
    assert "</script></div>" in page and "B <\\/script>" in page
    assert "<title>T &lt;b&gt;</title>" in page
    assert config.MARKMAP_AUTOLOADER_URL in page


def test_outline_round_trip():
    tree = mm.to_tree(OUTLINE)
    assert mm.to_tree(mm.to_outline(tree)) == tree


# ---------------------------------------------------------------- helpers
def test_link_timestamps():
    out = link_timestamps("See **[01:05]** and [1:00:00], not [x] or [00:10](done)", "VID")
    assert "**[01:05](https://www.youtube.com/watch?v=VID&t=65s)**" in out
    assert "[1:00:00](https://www.youtube.com/watch?v=VID&t=3600s)" in out
    assert "[x]" in out and "[00:10](done)" in out
    assert link_timestamps("[01:05]", None) == "[01:05]"


TRANSCRIPT = "\n".join(
    f"[{i // 2:02d}:{30 * (i % 2):02d}] " + text for i, text in enumerate([
        "welcome to the course on cooking pasta",
        "first boil plenty of salted water",
        "add the spaghetti and stir often",
        "now let us talk about tomato sauce",
        "garlic and olive oil go in the pan",
        "simmer the tomatoes for twenty minutes",
        "finally mix the pasta with the sauce",
        "serve with fresh basil and parmesan",
    ]))


def test_passages_and_search():
    passages = make_passages(TRANSCRIPT, lines_per_passage=2, overlap=0)
    assert len(passages) == 4 and passages[1].start == 60
    index = BM25Index(passages)
    best = index.search("how long do I simmer the tomatoes?", k=1)[0][1]
    assert "simmer the tomatoes" in best.text
    assert index.search("quantum physics", k=3) == []
    assert "the" not in tokenize("The sauce")


# ---------------------------------------------------------------- chat agent
@pytest.fixture
def captured(monkeypatch):
    calls = []

    def fake_ask(messages, **_):
        calls.append(messages)
        return "Simmer for twenty minutes [02:30]."
    monkeypatch.setattr(client, "ask", fake_ask)
    return calls


def _source():
    return chat_agent.ChatSource(title="Pasta class", transcript_text=TRANSCRIPT,
                                 notes={"short_summary": "How to cook pasta.",
                                        "long_summary": "LONG", "mind_map": "- Pasta"},
                                 duration=240)


def test_chat_full_context_with_memory(captured):
    history = [{"role": "user", "content": "What is this about?"},
               {"role": "assistant", "content": "Cooking pasta."}]
    reply = chat_agent.answer("How long to simmer?", history, _source())
    assert reply.mode == "full" and reply.answer.endswith("[02:30].")
    msgs = captured[0]
    system = msgs[0].content
    assert "Pasta class" in system and "04:00" in system
    assert "serve with fresh basil" in system           # whole transcript
    assert "How to cook pasta." in system and "LONG" not in system   # notes subset
    assert [m.type for m in msgs[1:]] == ["human", "ai", "human"]
    assert msgs[-1].content == "How long to simmer?"


def test_chat_search_mode_for_long_transcripts(captured, monkeypatch):
    monkeypatch.setattr(config, "CHAT_FULL_CONTEXT_MAX_CHARS", 100)
    monkeypatch.setattr(config, "CHAT_SEARCH_PASSAGES", 1)
    reply = chat_agent.answer("How long should the tomatoes simmer?", [], _source())
    assert reply.mode == "search" and len(reply.passages) == 1
    system = captured[0][0].content
    assert "simmer the tomatoes" in system
    assert "welcome to the course" not in system        # only the matching passage
    assert "EXCERPTS" in system


def test_chat_history_is_trimmed(captured, monkeypatch):
    monkeypatch.setattr(config, "CHAT_HISTORY_TURNS", 1)
    history = [{"role": r, "content": f"{r} {i}"} for i in range(5)
               for r in ("user", "assistant")]
    chat_agent.answer("Next?", history, _source())
    assert [m.content for m in captured[0][1:]] == ["user 4", "assistant 4", "Next?"]


def test_chat_rejects_empty_input(captured):
    with pytest.raises(InvalidInput):
        chat_agent.answer("   ", [], _source())
    with pytest.raises(InvalidInput):
        chat_agent.answer("Hi", [], chat_agent.ChatSource("t", ""))
    assert captured == []
