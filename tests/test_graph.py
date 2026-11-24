"""End-to-end tests of the agent graph with a fake LLM (no API calls)."""
import pytest

import config
from agents import transcription_agent
from agents.critic_agent import Critique, OutputReview
from agents.registry import ANALYSIS_AGENTS
from errors import TranscriptError
from graph.builder import get_graph
from llm import client

GOOD_MIND_MAP = ("- Topics overview\n  - Topic zero\n    - Sentences 0-4\n"
                 "  - Topic one\n    - Sentences 5-9\n  - Topic two\n    - Sentences 10-14")


def _segments(n=20):
    return [{"text": f"sentence number {i} about topic {i // 5}", "start": i * 6.0,
             "duration": 6.0} for i in range(n)]


class FakeLLM:
    """Answers like each agent would.

    fail_times: the critic fails the short summary this many times.
    bad_mind_map: the first mind map draft is prose instead of an outline.
    """

    def __init__(self, fail_times=0, bad_mind_map=False):
        self.fail_times = fail_times
        self.bad_mind_map = bad_mind_map
        self.calls = []

    def _answer(self, prompt) -> str:
        text = str(prompt)
        if "Translator Agent" in text:
            self.calls.append("translator")
            return "TRANSLATED"
        if "Chunk-Notes Agent" in text:
            self.calls.append("chunk_notes")
            return "- [00:00] notes"
        for agent in ANALYSIS_AGENTS.values():
            if f"You are the {agent.role} " in text:
                revising = "PREVIOUS DRAFT" in text
                self.calls.append(agent.role + (" (revise)" if revising else ""))
                if agent.key == "timeline":
                    return "- **[00:00]** Intro: start\n- **[01:00]** Topic: middle"
                if agent.key == "mind_map":
                    if self.bad_mind_map and not revising:
                        return "The video talks about three topics in order."
                    return GOOD_MIND_MAP
                return f"{agent.role} output"
        raise AssertionError(f"unexpected prompt: {text[:200]}")

    def ask(self, prompt, **_):
        return self._answer(prompt)

    def ask_many(self, prompts, **_):
        return [self._answer(p) for p in prompts]

    def ask_structured(self, schema, prompt, **_):
        assert schema is Critique
        self.calls.append("critic")
        text = str(prompt)
        reviews = []
        for k in ANALYSIS_AGENTS:
            if f"OUTPUT: {k}" in text:
                fail = k == "short_summary" and self.fail_times > 0
                reviews.append(OutputReview(output=k, passed=not fail,
                                            issues=["misses the ending"] if fail else []))
        self.fail_times -= 1
        return Critique(reviews=reviews)


@pytest.fixture
def fake_llm(monkeypatch, tmp_path):
    def make(**kw):
        fake = FakeLLM(**kw)
        monkeypatch.setattr(client, "ask", fake.ask)
        monkeypatch.setattr(client, "ask_many", fake.ask_many)
        monkeypatch.setattr(client, "ask_structured", fake.ask_structured)
        return fake
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    return make


def _run(**state):
    state.setdefault("options", {})
    return get_graph().invoke(state)


def test_default_outputs_with_one_revision(fake_llm, monkeypatch):
    fake = fake_llm(fail_times=1)
    monkeypatch.setattr(transcription_agent, "fetch_transcript", lambda *a: _segments())
    out = _run(source_type="youtube", url="https://youtu.be/dQw4w9WgXcQ")

    assert out["transcript_source"] == "captions"
    assert list(out["final_outputs"]) == ["short_summary", "long_summary", "mind_map",
                                          "takeaways"]
    assert out["revision_count"] == 1
    assert all(r["passed"] for r in out["reviews"].values())
    assert "Short Summary Agent (revise)" in fake.calls
    assert "Long Summary Agent (revise)" not in fake.calls
    assert fake.calls.count("critic") == 2
    assert "Timeline Agent" not in fake.calls   # not selected by default
    md = out["markdown"]
    assert "https://www.youtube.com/watch?v=dQw4w9WgXcQ" in md
    for title in ("## Short summary", "## Long summary", "## Mind map", "## Key takeaways"):
        assert title in md
    assert "```mermaid\nmindmap\n  root((Topics overview))" in md
    assert "translator" not in fake.calls


def test_malformed_mind_map_is_sent_back(fake_llm, monkeypatch):
    """The LLM critic passes everything, but the code check catches the mind map."""
    fake = fake_llm(bad_mind_map=True)
    monkeypatch.setattr(transcription_agent, "fetch_transcript", lambda *a: _segments())
    out = _run(source_type="youtube", url="dQw4w9WgXcQ",
               options={"outputs": ["mind_map", "takeaways"]})
    assert "Mind Map Agent (revise)" in fake.calls
    assert out["final_outputs"]["mind_map"] == GOOD_MIND_MAP
    assert out["reviews"]["mind_map"]["passed"]
    assert out["revision_count"] == 1


def test_revisions_stop_at_max(fake_llm, monkeypatch):
    fake_llm(fail_times=99)
    monkeypatch.setattr(transcription_agent, "fetch_transcript", lambda *a: _segments())
    out = _run(source_type="youtube", url="dQw4w9WgXcQ")
    assert out["revision_count"] == config.MAX_REVISIONS
    assert not out["reviews"]["short_summary"]["passed"]
    assert out["markdown"]  # still produces notes


def test_missing_captions_fall_back_to_speech_to_text(fake_llm, monkeypatch, tmp_path):
    fake_llm()

    def no_captions(*_):
        raise TranscriptError("Captions are disabled for this video.", "disabled")

    audio = tmp_path / "a.webm"
    audio.write_bytes(b"x")
    calls = []
    monkeypatch.setattr(transcription_agent, "fetch_transcript", no_captions)
    monkeypatch.setattr(transcription_agent, "download_youtube_audio",
                        lambda vid, d: (audio, {"title": "My video", "duration": 120}))

    def fake_transcribe(path, work_dir, **kw):
        calls.append(kw["engine"])
        return _segments(), {"language": "en", "engine": kw["engine"]}

    monkeypatch.setattr(transcription_agent, "transcribe", fake_transcribe)
    opts = {"outputs": ["short_summary"]}
    out = _run(source_type="youtube", url="dQw4w9WgXcQ", options=opts)

    assert out["transcript_source"] == "whisper"
    assert out["caption_problem"] == "Captions are disabled for this video."
    assert out["title"] == "My video"
    assert list(out["final_outputs"]) == ["short_summary"]

    # second run reuses the cached transcript: no download, no transcription
    monkeypatch.setattr(transcription_agent, "download_youtube_audio",
                        lambda *a: pytest.fail("should use the cache"))
    out2 = _run(source_type="youtube", url="dQw4w9WgXcQ", options=opts)
    assert out2["segments"] == out["segments"]
    assert calls == ["whisper"]


def test_fallback_disabled_raises(fake_llm, monkeypatch):
    fake_llm()

    def no_captions(*_):
        raise TranscriptError("No captions are available for this video.", "not_found")

    monkeypatch.setattr(transcription_agent, "fetch_transcript", no_captions)
    with pytest.raises(TranscriptError, match="speech-to-text fallback"):
        _run(source_type="youtube", url="dQw4w9WgXcQ",
             options={"allow_stt_fallback": False})


def test_unavailable_video_does_not_fall_back(fake_llm, monkeypatch):
    fake_llm()

    def unavailable(*_):
        raise TranscriptError("This video is unavailable.", "unavailable")

    monkeypatch.setattr(transcription_agent, "fetch_transcript", unavailable)
    monkeypatch.setattr(transcription_agent, "download_youtube_audio",
                        lambda *a: pytest.fail("should not download"))
    with pytest.raises(TranscriptError, match="unavailable"):
        _run(source_type="youtube", url="dQw4w9WgXcQ")


def test_upload_long_recording_uses_map_reduce_and_translation(fake_llm, monkeypatch, tmp_path):
    fake = fake_llm()
    monkeypatch.setattr(config, "SINGLE_PASS_MAX_CHARS", 300)
    monkeypatch.setattr(config, "CHUNK_MAX_CHARS", 200)
    f = tmp_path / "lecture.mp3"
    f.write_bytes(b"fake audio")
    monkeypatch.setattr(transcription_agent, "transcribe",
                        lambda *a, **kw: (_segments(40), {"language": "hi", "engine": "gemini"}))

    out = _run(source_type="upload", file_path=str(f), file_name="lecture.mp3",
               options={"language": "Hindi", "stt_engine": "gemini",
                        "outputs": ["timeline", "long_summary"], "use_critic": False})

    assert out["needs_map_reduce"] and out["source_kind"] == "condensed notes"
    assert "chunk_notes" in fake.calls
    assert out["transcript_source"] == "gemini"
    assert out["final_outputs"] == {"timeline": "TRANSLATED", "long_summary": "TRANSLATED"}
    assert "critic" not in fake.calls
    assert "Notes: lecture" in out["markdown"]
    assert f.exists()  # the user's file is never deleted
