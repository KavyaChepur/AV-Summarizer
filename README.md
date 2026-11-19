# AV Summarizer AI (agentic)

A multi-agent application that turns **YouTube videos** and **audio/video files** into study notes: a summary, key takeaways and timestamped notes. It is built with **LangGraph** (the agent graph) and **LangChain** (prompts and the Gemini model), with a Streamlit web interface and a command-line version.

- **YouTube links:** the app uses the video's captions. If there are none, it downloads the audio and transcribes it.
- **Uploaded files:** MP3, WAV, M4A, MP4, MKV, WEBM and more are transcribed with speech-to-text.
- **Speech-to-text engines:** *Whisper* (faster-whisper, runs locally and free) or *Gemini* (cloud).
- **A critic agent** checks every output against the transcript and sends flawed drafts back to the agent that wrote them.
- **Output languages:** English, Hindi, Telugu, Spanish, French and German, handled by a translator agent.
- **Export** as Markdown, text or PDF, plus the full transcript.

---

## Contents

1. [Quick start](#quick-start)
2. [Architecture](#architecture)
3. [The agents](#the-agents)
4. [Project structure](#project-structure)
5. [Configuration](#configuration)
6. [Command-line use](#command-line-use)
7. [Tests](#tests)
8. [Gemini quotas](#gemini-quotas)
9. [Limitations](#limitations)
10. [Ideas for extending it](#ideas-for-extending-it)

---

## Quick start

**Requirements:** Python 3.10+, `ffmpeg`, and a Google Gemini API key from [Google AI Studio](https://aistudio.google.com/).

```bash
cd av-summarizer
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
sudo apt install ffmpeg                               # macOS: brew install ffmpeg

cp .env.example .env                                  # then put your key in .env
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501), paste a YouTube link or upload a file, and click **Generate notes**. The live log shows each agent as it works.

The first time Whisper is used, it downloads its model (about 75 MB for `base`, 480 MB for `small`).

---

## Architecture

```
                    ┌──────────────────────────────────────┐
                    │  Streamlit UI (app.py) / CLI          │
                    │  YouTube link  |  file upload         │
                    └──────────────────┬───────────────────┘
                                       │ initial state + options
                                       ▼
 ┌────────────────────────── LangGraph StateGraph ──────────────────────────┐
 │                                                                          │
 │  ingest ──(upload)──────────────────────────────────┐                    │
 │    │                                                 │                    │
 │ (youtube)          (always speech-to-text)           │                    │
 │    ├──────────────────────────────┐                  │                    │
 │    ▼                              ▼                  ▼                    │
 │  fetch_captions ──(no captions)─► download_audio ──► speech_to_text        │
 │    │ (ok)                          │ (cached transcript)  │                 │
 │    └───────────────┬───────────────┘◄────────────────────┘                 │
 │                    ▼                                                       │
 │               preprocess ──(long recording)──► chunk_notes (map-reduce)    │
 │                    │                                │                      │
 │                    └────────── fan-out ─────────────┘                      │
 │                 ┌──────────────┼──────────────┐       (in parallel)        │
 │                 ▼              ▼              ▼                            │
 │              summary       takeaways      timeline                         │
 │                 └──────────────┼──────────────┘                            │
 │                                ▼                                           │
 │                             critic ──(issues, < MAX_REVISIONS)──► revise ─┐│
 │                                │  ▲                                       ││
 │                                │  └───────────────────────────────────────┘│
 │                    (not English)│(English)                                  │
 │                  translate ◄───┴───► export ──► END                        │
 └──────────────────────────────────────────────────────────────────────────┘
```

`python run_cli.py --graph` prints the real compiled graph as a Mermaid diagram.

### How the pieces fit together

- **Shared state** ([graph/state.py](graph/state.py)): one `PipelineState` dictionary flows through the graph. Each node reads what it needs and returns only the keys it changes. The `log` list uses a LangGraph *reducer* (`operator.add`), so agents running in parallel can all add to it.
- **Orchestrator** ([graph/router.py](graph/router.py)): the routing decisions. These are plain, rule-based functions: captions or speech-to-text, single pass or map-reduce, which writers to start, whether to revise, whether to translate. They're kept as rules on purpose, because these decisions have clear right answers and rules are fast, free and predictable.
- **Agents vs. tools:** a node is an *agent* when an LLM has to think (write, judge, translate). Steps with one right answer (download, convert, chunk, export) are deterministic *tools* in [tools/](tools/).
- **One LLM gateway** ([llm/client.py](llm/client.py)): every Gemini call goes through `ask`, `ask_many` (parallel) or `ask_structured` (Pydantic output). It handles retries, waits out per-minute rate limits, and turns API errors into readable messages. To switch to another provider (e.g. Ollama), only `get_llm` needs to change.
- **Live progress:** nodes send progress events through LangGraph's stream writer (`stream_mode="custom"`). [graph/runner.py](graph/runner.py) turns the stream into simple events for the UI and CLI.

### Audio pipeline

```
uploaded file / downloaded YouTube audio (yt-dlp)
        │
        ▼  ffmpeg → 16 kHz mono WAV
        │
        ├── Whisper: faster-whisper on the whole file, with voice-activity
        │            detection; reports progress as it goes
        │
        └── Gemini:  cut into 10-minute mp3 pieces with a 10 s overlap → each piece
                     is sent to Gemini with a structured-output schema → timestamps are
                     shifted to absolute time and overlap duplicates are removed
        │
        ▼
segments = [{"text", "start", "duration"}, ...]   ← the same format as YouTube captions
```

Because speech-to-text produces exactly the same segment format as captions, everything after transcription works the same for both sources. Transcripts are cached in `.cache/transcripts/` (by video ID, or by the file's hash), so the same audio is never transcribed twice.

---

## The agents

| Node | Type | What it does | File |
|---|---|---|---|
| `ingest` | Ingestion agent | Checks the link or file and works out the source type | [agents/ingestion_agent.py](agents/ingestion_agent.py) |
| `fetch_captions` | Transcription agent | Gets YouTube captions. If they're missing, it decides whether speech-to-text can help (yes for missing or disabled captions, no for a private video) | [agents/transcription_agent.py](agents/transcription_agent.py) |
| `download_audio` | Transcription agent | Downloads the audio with yt-dlp, or reuses a cached transcript | same |
| `speech_to_text` | Transcription agent | Whisper or Gemini transcription | same |
| `preprocess` | Tool | Cleans the text, builds `[MM:SS]` lines, chunks it, and chooses single pass or map-reduce | [agents/preprocess.py](agents/preprocess.py) |
| `chunk_notes` | Chunk-notes agent (LLM) | Map: notes for every chunk, in parallel. Reduce: merges them until they fit | [agents/chunk_notes_agent.py](agents/chunk_notes_agent.py) |
| `summary` | Summary agent (LLM) | Short / Medium / Detailed summary | [agents/summary_agent.py](agents/summary_agent.py) |
| `takeaways` | Takeaways agent (LLM) | 5–8 key takeaways | [agents/takeaways_agent.py](agents/takeaways_agent.py) |
| `timeline` | Timeline agent (LLM) | Chronological `[MM:SS]` notes | [agents/timeline_agent.py](agents/timeline_agent.py) |
| `critic` | Critic agent (LLM + code) | An LLM fact-check with structured output, plus a code check that timeline timestamps exist in the source, are in order and are within the recording's length | [agents/critic_agent.py](agents/critic_agent.py) |
| `revise` | the writers again | Sends each failed output, with the critic's issues, back to the agent that wrote it | same |
| `translate` | Translator agent (LLM) | Translates the final outputs and keeps the Markdown and timestamps | [agents/translator_agent.py](agents/translator_agent.py) |
| `export` | Tool | Builds the final Markdown document | [agents/export_agent.py](agents/export_agent.py) |

The three writer agents share [agents/analysis_base.py](agents/analysis_base.py). Each can write a first draft and revise its own draft from feedback. They always write in English and leave translation to the translator agent, so the critic always compares English notes with the source.

---

## Project structure

```
av-summarizer/
├── app.py                  # Streamlit UI
├── run_cli.py              # command-line version
├── config.py               # all settings (read from .env)
├── errors.py               # exceptions with user-friendly messages
├── graph/
│   ├── state.py            # PipelineState (shared state)
│   ├── router.py           # orchestrator: routing decisions
│   ├── builder.py          # builds and compiles the LangGraph graph
│   └── runner.py           # streams the graph as simple events
├── agents/                 # one file per agent (see the table above)
├── tools/
│   ├── youtube_tools.py    # link parsing, captions
│   ├── audio_tools.py      # yt-dlp download, ffmpeg convert/split/merge
│   ├── stt_tools.py        # Whisper and Gemini speech-to-text
│   ├── text_tools.py       # cleaning, timestamps, chunking
│   ├── export_tools.py     # Markdown and PDF
│   └── cache.py            # transcript cache
├── llm/client.py           # the only code that calls Gemini (LangChain)
├── tests/                  # pytest: tools, graph (fake LLM), LLM client, UI
├── legacy/                 # the original single-pipeline version, for reference
├── .streamlit/config.toml  # allows uploads up to 1 GB
├── requirements.txt
└── .env.example
```

---

## Configuration

Set these in `.env`. Only `GOOGLE_API_KEY` is required.

| Variable | Default | Meaning |
|---|---|---|
| `GOOGLE_API_KEY` | (required) | Gemini API key |
| `GEMINI_MODEL` | `gemini-3.5-flash` | Model used by all agents |
| `GEMINI_FALLBACK_MODELS` | `gemini-2.5-flash,gemini-3.1-flash-lite,gemini-flash-latest,gemini-2.5-flash-lite` | Comma-separated models tried in order when the main one is overloaded, out of quota or unavailable |
| `STT_ENGINE` | `whisper` | Default speech-to-text engine (`whisper` or `gemini`) |
| `WHISPER_MODEL` | `base` | `tiny`, `base`, `small` or `medium`. Bigger is more accurate but slower |
| `AUDIO_PIECE_SECONDS` | `600` | Gemini speech-to-text: length of each audio piece |
| `AUDIO_OVERLAP_SECONDS` | `10` | Overlap between pieces |
| `SINGLE_PASS_MAX_CHARS` | `60000` | Transcripts up to this length skip map-reduce |
| `CHUNK_MAX_CHARS` | `24000` | Chunk size for the map step |
| `REDUCE_MAX_CHARS` | `40000` | Target size of the condensed notes |
| `MAP_CONCURRENCY` | `4` | Parallel Gemini calls in the map step |
| `MAX_REVISIONS` | `2` | Maximum critic → revise rounds |
| `TIMESTAMP_TOLERANCE_SECONDS` | `60` | How far a timeline timestamp may be from a source timestamp |
| `AVS_CACHE_DIR` | `.cache` | Where transcripts are cached |

The sidebar in the app sets the per-run options: summary length, outputs, language, speech-to-text engine and Whisper model, the YouTube speech-to-text options, and whether the critic runs.

**Which Whisper model?** On a CPU, `base` takes a few seconds per minute of audio and is fine for clear speech. Use `small` for better accuracy, especially for Indian languages or accented speech. `medium` is best but slow without a GPU.

---

## Command-line use

```bash
python run_cli.py "https://youtu.be/VIDEO_ID"                       # notes to stdout
python run_cli.py lecture.mp3 --length Detailed -o notes.md          # uploaded file
python run_cli.py "https://youtu.be/VIDEO_ID" --force-stt --stt-engine gemini
python run_cli.py talk.mp4 --language Hindi --outputs summary,timeline
python run_cli.py --graph                                            # Mermaid diagram
```

Run `python run_cli.py -h` for all options.

---

## Tests

```bash
python -m pytest
```

The tests make **no API calls**:

- [tests/test_tools.py](tests/test_tools.py): link parsing, timestamps, chunking, the critic's timestamp check, audio split/merge (uses a real ffmpeg file).
- [tests/test_graph.py](tests/test_graph.py): the whole agent graph with a fake LLM. It covers captions with a critic revision, the revision limit, the caption → speech-to-text fallback and the transcript cache, the fallback switched off, private videos, and map-reduce with translation.
- [tests/test_llm_client.py](tests/test_llm_client.py): rate-limit waiting, daily-quota and unknown-model errors.
- [tests/test_app.py](tests/test_app.py): the Streamlit UI (Streamlit `AppTest`) and PDF export.

---

## Gemini quotas

On the free tier, Gemini limits both **requests per minute** and **requests per day** for each model.

- A short video with captions needs about **5 calls**: 3 writers + 1 critic, plus 3 translations and any revisions.
- Long recordings need one extra call per ~24k-character chunk, and Gemini speech-to-text needs one per 10 minutes of audio.
- **Per-minute limits** are handled automatically: the app waits and retries, and shows "waiting…".
- **Overloaded models (503)** and **daily limits** make the app switch to the next model in `GEMINI_FALLBACK_MODELS` (each model has its own quota and load). It stays on the model that worked for 10 minutes, then tries the main model again. If every model is overloaded, it waits 20 s and tries the list again, up to 3 times. The run stops only if every model fails.

To use fewer calls, transcribe with Whisper instead of Gemini, and turn off the critic for quick drafts.

---

## Limitations

- **Accuracy depends on the transcript.** YouTube auto-captions and small Whisper models make mistakes (for example "trunks" heard as "punks"). The critic checks the notes against the transcript, not against the audio.
- **Gemini speech-to-text timestamps** are approximate (to within a few seconds). Use Whisper when timestamps matter.
- **YouTube may block downloads** from cloud servers or VPNs, for both captions and audio. Age-restricted and private videos can't be fetched.
- **PDF export** isn't offered for Hindi and Telugu, because the PDF library can't lay out those scripts correctly. Use the Markdown download instead.
- Whisper needs CPU time and RAM: `medium` needs about 2 GB of RAM and is slow on a CPU.

---

## Ideas for extending it

- **Q&A agent:** chat with the video using the transcript (the transcript is already in the state).
- **Clickable timestamps** that open the video at that moment (`&t=SECONDS`).
- **Speaker labels** (diarization) for interviews and meetings.
- **Local LLM:** replace `get_llm` with `ChatOllama` to run completely offline.
- **LangGraph checkpointing** (`langgraph-checkpoint-sqlite` is installed) to resume long runs and keep a history of past notes.
- **Tracing** with LangSmith to inspect every agent's prompt and answer.

---

## Security

Never commit `.env`; it contains your API key. The included `.gitignore` already excludes it, along with the transcript cache.
