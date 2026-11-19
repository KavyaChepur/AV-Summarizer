"""Smoke tests of the Streamlit UI with a fake pipeline (no API calls)."""
import pytest
from streamlit.testing.v1 import AppTest

import graph.runner
from errors import TranscriptError

FINAL = {
    "final_outputs": {"summary": "A **short** summary.",
                      "takeaways": "1. One\n2. Two",
                      "timeline": "- **[00:00]** Intro: hello"},
    "markdown": "# Notes: Test\n\n## Summary\n\nA **short** summary.\n- bullet\n1. one\n",
    "transcript_text": "[00:00] hello world",
    "reviews": {"summary": {"passed": False, "issues": ["misses the ending"]},
                "takeaways": {"passed": True, "issues": []},
                "timeline": {"passed": True, "issues": []}},
    "transcript_source": "captions", "duration": 19.0, "detected_language": None,
    "caption_problem": None, "video_id": "jNQXAC9IVRw", "title": None,
}


def fake_stream(state):
    yield {"type": "progress", "agent": "summary", "message": "writing", "progress": None}
    yield {"type": "step", "node": "ingest", "log": ["Recognised video."]}
    yield {"type": "done", "state": {**state, **FINAL}}


def failing_stream(state):
    raise TranscriptError("Captions are disabled for this video.", "disabled")
    yield  # pragma: no cover


def _app(monkeypatch, stream):
    monkeypatch.setattr(graph.runner, "stream_pipeline", stream)
    return AppTest.from_file("app.py", default_timeout=30).run()


def test_initial_page(monkeypatch):
    at = _app(monkeypatch, fake_stream)
    assert not at.exception
    assert at.button[0].disabled  # nothing entered yet


def test_invalid_link_shows_error(monkeypatch):
    at = _app(monkeypatch, fake_stream)
    at.text_input[0].input("https://vimeo.com/1").run()
    assert "doesn't look like a YouTube link" in at.error[0].value
    assert at.button[0].disabled


def test_full_run_shows_results(monkeypatch):
    at = _app(monkeypatch, fake_stream)
    at.text_input[0].input("https://youtu.be/jNQXAC9IVRw").run()
    assert not at.button[0].disabled
    at.button[0].click().run()
    assert not at.exception
    labels = [t.label for t in at.tabs]
    assert labels == ["Summary", "Key takeaways", "Timestamped notes", "Transcript",
                      "Agent trace"]
    assert any("critic agent still has concerns" in w.value for w in at.warning)


def test_pipeline_error_is_shown(monkeypatch):
    at = _app(monkeypatch, failing_stream)
    at.text_input[0].input("jNQXAC9IVRw").run()
    at.button[0].click().run()
    assert not at.exception
    assert any("Captions are disabled" in e.value for e in at.error)


def test_pdf_export():
    from tools.export_tools import markdown_to_pdf
    pdf = markdown_to_pdf(FINAL["markdown"] + "\n<odd & chars>\n### h3\n  - nested\n")
    assert pdf.startswith(b"%PDF")
