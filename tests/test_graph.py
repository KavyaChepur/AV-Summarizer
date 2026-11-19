"""End-to-end tests of the agent graph with a fake LLM (no API calls)."""
import pytest

import config
from agents import transcription_agent
from agents.critic_agent import Critique, OutputReview
from errors import TranscriptError
from graph.builder import get_graph
from llm import client


def _segments(n=20):
    return [{"text": f"sentence number {i} about topic {i // 5}", "start": i * 6.0,
             "duration": 6.0} for i in range(n)]


class FakeLLM:
    """Answers like each agent would; the critic fails the summary `fail_times` times."""

    def __init__(self, fail_times=0):
        self.fail_times = fail_times
        self.calls = []

    def _answer(self, prompt) -> str:
        text = str(prompt)
        if "Translator Agent" in text:
            self.calls.append("translator")
            return "TRANSLATED"
        if "Chunk-Notes Agent" in text:
            self.calls.append("chunk_notes")
            return "- [00:00] notes"
        for role in ("Summary Agent", "Takeaways Agent", "Timeline Agent"):
            if f"You are the {role}" in text:
                self.calls.append(role + (" (revise)" if "PREVIOUS DRAFT" in text else ""))
                if role == "Timeline Agent":
                    return "- **[00:00]** Intro: start\n- **[01:00]** Topic: middle"
                return f"{role} output"
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
        for k in ("summary", "takeaways", "timeline"):
            if f"OUTPUT: {k}" in text:
                fail = k == "summary" and self.fail_times > 0
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


def test_youtube_captions_with_one_revision(fake_llm, monkeypatch):
    fake = fake_llm(fail_times=1)
    monkeypatch.setattr(transcription_agent, "fetch_transcript", lambda *a: _segments())
    out = _run(source_type="youtube", url="https://youtu.be/dQw4w9WgXcQ")

    assert out["transcript_source"] == "captions"
    assert list(out["final_outputs"]) == ["summary", "takeaways", "timeline"]
    assert out["revision_count"] == 1
    assert all(r["passed"] for r in out["reviews"].values())
    assert "Summary Agent (revise)" in fake.calls
    assert fake.calls.count("critic") == 2
    assert "https://www.youtube.com/watch?v=dQw4w9WgXcQ" in out["markdown"]
    assert "translator" not in fake.calls


def test_revisions_stop_at_max(fake_llm, monkeypatch):
    fake_llm(fail_times=99)
    monkeypatch.setattr(transcription_agent, "fetch_transcript", lambda *a: _segments())
    out = _run(source_type="youtube", url="dQw4w9WgXcQ")
    assert out["revision_count"] == config.MAX_REVISIONS
    assert not out["reviews"]["summary"]["passed"]
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
    out = _run(source_type="youtube", url="dQw4w9WgXcQ", options={"outputs": ["summary"]})

    assert out["transcript_source"] == "whisper"
    assert out["caption_problem"] == "Captions are disabled for this video."
    assert out["title"] == "My video"
    assert list(out["final_outputs"]) == ["summary"]

    # second run reuses the cached transcript: no download, no transcription
    monkeypatch.setattr(transcription_agent, "download_youtube_audio",
                        lambda *a: pytest.fail("should use the cache"))
    out2 = _run(source_type="youtube", url="dQw4w9WgXcQ", options={"outputs": ["summary"]})
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
                        "outputs": ["timeline", "summary"], "use_critic": False})

    assert out["needs_map_reduce"] and out["source_kind"] == "condensed notes"
    assert "chunk_notes" in fake.calls
    assert out["transcript_source"] == "gemini"
    assert out["final_outputs"] == {"timeline": "TRANSLATED", "summary": "TRANSLATED"}
    assert "critic" not in fake.calls
    assert "Notes: lecture" in out["markdown"]
    assert f.exists()  # the user's file is never deleted
