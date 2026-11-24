"""Smoke tests of the Streamlit UI with a fake pipeline and chat (no API calls)."""
from streamlit.testing.v1 import AppTest

import graph.runner
from agents import chat_agent
from errors import LLMError, TranscriptError

MIND_MAP = "- Elephants\n  - Setting\n    - At the zoo\n  - Trunks\n    - Really long"
FINAL = {
    "final_outputs": {"short_summary": "A **short** summary.",
                      "long_summary": "### Part one\nDetails at [00:05].",
                      "mind_map": MIND_MAP,
                      "takeaways": "1. One\n2. Two"},
    "doc_title": "Notes: Test", "source_line": "Source: test",
    "transcript_note": "Transcript from YouTube captions",
    "markdown": "# Notes: Test\n\n## Short summary\n\nA **short** summary.\n",
    "transcript_text": "[00:00] hello world",
    "reviews": {"short_summary": {"passed": False, "issues": ["misses the ending"]},
                "long_summary": {"passed": True, "issues": []},
                "mind_map": {"passed": True, "issues": []},
                "takeaways": {"passed": True, "issues": []}},
    "transcript_source": "captions", "duration": 19.0, "detected_language": None,
    "caption_problem": None, "video_id": "jNQXAC9IVRw", "title": None,
}


def fake_stream(state):
    yield {"type": "progress", "agent": "short_summary", "message": "writing",
           "progress": None}
    yield {"type": "step", "node": "ingest", "log": ["Recognised video."]}
    yield {"type": "done", "state": {**state, **FINAL}}


def failing_stream(state):
    raise TranscriptError("Captions are disabled for this video.", "disabled")
    yield  # pragma: no cover


def _app(monkeypatch, stream):
    monkeypatch.setattr(graph.runner, "stream_pipeline", stream)
    return AppTest.from_file("app.py", default_timeout=30).run()


def _run_notes(monkeypatch):
    at = _app(monkeypatch, fake_stream)
    at.text_input[0].input("https://youtu.be/jNQXAC9IVRw").run()
    at.button[0].click().run()
    assert not at.exception
    return at


def test_initial_page(monkeypatch):
    at = _app(monkeypatch, fake_stream)
    assert not at.exception
    assert at.button[0].disabled  # nothing entered yet
    assert at.multiselect[0].value == ["Short summary", "Long summary", "Mind map",
                                       "Key takeaways"]


def test_invalid_link_shows_error(monkeypatch):
    at = _app(monkeypatch, fake_stream)
    at.text_input[0].input("https://vimeo.com/1").run()
    assert "doesn't look like a YouTube link" in at.error[0].value
    assert at.button[0].disabled


def test_full_run_shows_results(monkeypatch):
    at = _run_notes(monkeypatch)
    assert [t.label for t in at.tabs] == ["Short summary", "Long summary", "Mind map",
                                          "Key takeaways", "Transcript"]
    assert any("critic agent still has concerns" in w.value for w in at.warning)
    # timestamps become links to that moment in the video
    assert any("watch?v=jNQXAC9IVRw&t=5s" in m.value for m in at.markdown)
    assert len(at.chat_input) == 1


def test_chat_answers_and_remembers(monkeypatch):
    seen = []

    def fake_answer(question, history, source):
        seen.append((question, list(history), source.title))
        return chat_agent.ChatReply(f"Answer to '{question}' [00:00]", "full")

    monkeypatch.setattr(chat_agent, "answer", fake_answer)
    at = _run_notes(monkeypatch)
    at.chat_input[0].set_value("Why elephants?").run()
    assert not at.exception
    assert seen[0] == ("Why elephants?", [], "YouTube video jNQXAC9IVRw")
    assert [m.name for m in at.chat_message] == ["user", "assistant"]

    at.chat_input[0].set_value("And the trunks?").run()
    question, history, _ = seen[1]
    assert question == "And the trunks?"
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert len(at.chat_message) == 4

    clear = [b for b in at.button if b.label == "Clear chat"][0]
    clear.click().run()
    assert len(at.chat_message) == 0


def test_chat_suggestion_button_and_error(monkeypatch):
    def failing_answer(question, history, source):
        raise LLMError("Gemini quota exceeded")

    monkeypatch.setattr(chat_agent, "answer", failing_answer)
    at = _run_notes(monkeypatch)
    suggestion = [b for b in at.button if b.label == "What is the main idea?"][0]
    suggestion.click().run()
    assert not at.exception
    assert any("quota exceeded" in e.value for e in at.error)
    # a failed answer isn't kept in the history
    assert not [b for b in at.button if b.label == "Clear chat"]


def test_pipeline_error_is_shown(monkeypatch):
    at = _app(monkeypatch, failing_stream)
    at.text_input[0].input("jNQXAC9IVRw").run()
    at.button[0].click().run()
    assert not at.exception
    assert any("Captions are disabled" in e.value for e in at.error)


def test_pdf_export_with_mind_map():
    from tools.export_tools import markdown_to_pdf, notes_to_pdf
    pdf = markdown_to_pdf("# T\n<odd & chars>\n### h3\n  - nested\n```mermaid\nx\n```\n")
    assert pdf.startswith(b"%PDF")
    pdf = notes_to_pdf("Notes", "Source: x", FINAL["final_outputs"], "note")
    assert pdf.startswith(b"%PDF") and len(pdf) > 2000
