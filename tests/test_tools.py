import shutil
import subprocess

import pytest

from agents.critic_agent import validate_timestamps
from errors import InvalidYouTubeURL
from tools import audio_tools
from tools.audio_tools import AudioPiece, merge_piece_segments
from tools.text_tools import (chunk_transcript, clean_text, find_timestamps, fmt_time,
                              parse_time)
from tools.youtube_tools import extract_video_id

VID = "dQw4w9WgXcQ"


@pytest.mark.parametrize("url", [
    VID,
    f"https://www.youtube.com/watch?v={VID}",
    f"https://www.youtube.com/watch?feature=share&v={VID}&t=30",
    f"youtube.com/watch?v={VID}",
    f"https://youtu.be/{VID}?si=abc",
    f"https://www.youtube.com/shorts/{VID}",
    f"https://www.youtube.com/embed/{VID}",
    f"https://www.youtube.com/live/{VID}",
    f"https://m.youtube.com/watch?v={VID}",
    f"https://music.youtube.com/watch?v={VID}",
])
def test_extract_video_id(url):
    assert extract_video_id(url) == VID


@pytest.mark.parametrize("url", ["", "   ", "https://vimeo.com/123",
                                 "https://www.youtube.com/watch?v=short",
                                 "https://www.youtube.com/channel/abc"])
def test_extract_video_id_rejects(url):
    with pytest.raises(InvalidYouTubeURL):
        extract_video_id(url)


def test_time_helpers():
    assert fmt_time(75) == "01:15"
    assert fmt_time(3723) == "1:02:03"
    assert parse_time("12:30") == 750
    assert parse_time("1:02:03") == 3723
    with pytest.raises(ValueError):
        parse_time("12")
    assert find_timestamps("- **[00:30]** a\n- [1:02:03] b [x]") == [30, 3723]


def test_clean_text():
    assert clean_text("[Music] hello &amp;   world [Applause]") == "hello & world"


def _segments(n, seconds=5.0, words="word " * 20):
    return [{"text": words, "start": i * seconds, "duration": seconds} for i in range(n)]


def test_chunk_transcript_respects_limit_and_covers_everything():
    segs = _segments(200)
    chunks = chunk_transcript(segs, max_chars=2000)
    assert len(chunks) > 1
    assert all(len(c.text) <= 2000 + 700 for c in chunks)  # one ~30 s line of slack
    assert chunks[0].start == 0
    assert chunks[-1].end == pytest.approx(1000)
    assert all(a.end == b.start for a, b in zip(chunks, chunks[1:]))


def test_validate_timestamps():
    source = "[00:00] a\n[00:30] b\n[01:00] c\n[05:00] d"
    assert validate_timestamps("- [00:00] x\n- [01:00] y", source, 330) == []
    issues = validate_timestamps("- [00:00] x\n- [03:00] y\n- [10:00] z", source, 330)
    assert any("later than the end" in i for i in issues)
    assert any("don't match" in i for i in issues)
    assert validate_timestamps("- [01:00] x\n- [00:00] y", source, 330) == [
        "The timestamps are not in chronological order."]
    assert validate_timestamps("no stamps", source, 330)


def test_merge_piece_segments_drops_overlap_duplicates():
    # pieces of 600 s with 10 s overlap: piece 1 starts at 590 and owns >= 595
    pieces = [AudioPiece(None, 0, 0, 595), AudioPiece(None, 590, 595, float("inf"))]
    p0 = [{"text": "a", "start": 0, "duration": 5},
          {"text": "b", "start": 592, "duration": 5},    # 592 < 595: kept from piece 0
          {"text": "c", "start": 597, "duration": 3}]    # owned by piece 1: dropped
    p1 = [{"text": "b", "start": 2, "duration": 5},      # 592 absolute: dropped
          {"text": "c", "start": 7, "duration": 3},      # 597 absolute: kept
          {"text": "d", "start": 20, "duration": 3}]
    merged = merge_piece_segments(pieces, [p0, p1])
    assert [s["text"] for s in merged] == ["a", "b", "c", "d"]
    assert merged[2]["start"] == 597


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_split_audio_real_file(tmp_path):
    src = tmp_path / "tone.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=25", str(src)], check=True)
    pieces = audio_tools.split_audio(src, tmp_path, piece_seconds=10, overlap=2)
    assert len(pieces) == 3
    assert [p.offset for p in pieces] == [0, 8, 18]
    assert pieces[0].keep_from == 0 and pieces[-1].keep_to == float("inf")
    assert all(p.path.exists() for p in pieces)
    assert audio_tools.probe_duration(pieces[1].path) == pytest.approx(12, abs=0.3)
