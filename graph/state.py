"""The shared state every agent reads from and writes to."""
import operator
from typing import Annotated, Dict, List, Literal, Optional, TypedDict

from tools.text_tools import Chunk


class Options(TypedDict, total=False):
    length: str                 # "Short" | "Medium" | "Detailed"
    outputs: List[str]          # any of "summary", "takeaways", "timeline", in display order
    language: str               # output language, e.g. "English"
    stt_engine: str             # "whisper" | "gemini"
    whisper_model: str          # "tiny" | "base" | "small" | "medium"
    force_stt: bool             # YouTube: skip captions and transcribe the audio
    allow_stt_fallback: bool    # YouTube: transcribe the audio if captions are missing
    use_critic: bool            # run the critic/revision loop


class Review(TypedDict):
    passed: bool
    issues: List[str]


class PipelineState(TypedDict, total=False):
    # ---- input
    source_type: Literal["youtube", "upload"]
    url: Optional[str]
    file_path: Optional[str]
    file_name: Optional[str]
    options: Options

    # ---- ingestion
    video_id: Optional[str]
    source_key: str             # stable id of the input, used for caching
    title: Optional[str]
    work_dir: Optional[str]     # temp folder for downloaded/converted audio
    audio_path: Optional[str]

    # ---- transcription
    segments: List[Dict]        # [{text, start, duration}]
    transcript_source: str      # "captions" | "whisper" | "gemini"
    detected_language: Optional[str]
    caption_problem: Optional[str]   # why captions were not used, if they weren't

    # ---- preprocessing
    duration: float
    transcript_text: str        # readable "[MM:SS] …" transcript
    chunks: List[Chunk]
    needs_map_reduce: bool
    source_text: str            # what the analysis agents read
    source_kind: str            # "transcript" | "condensed notes"

    # ---- analysis (English drafts)
    summary: str
    takeaways: str
    timeline: str

    # ---- quality control
    reviews: Dict[str, Review]
    revision_count: int

    # ---- output
    final_outputs: Dict[str, str]
    markdown: str

    # ---- trace: every agent appends what it did (parallel-safe)
    log: Annotated[List[str], operator.add]
