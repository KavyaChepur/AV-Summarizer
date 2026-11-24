# AV Summarizer AI (agentic)

A multi-agent application that turns **YouTube videos** and **audio/video files** into study notes, and lets you **chat with the recording** afterwards. It is built with **LangGraph** (the agent graph) and **LangChain** (prompts and the Gemini model), with a Streamlit web interface and a command-line version.

**Outputs:**

| Output | What you get |
|---|---|
| **Short summary** | The whole recording in 80–120 words |
| **Long summary** | 600–900 words with a heading per topic, in the order they're discussed |
| **Mind map** | An interactive, collapsible mind map in the app; also downloadable as a web page, drawn in the PDF, and included as a Mermaid diagram in the Markdown notes |
| **Key takeaways** | The 5–8 most important points |
| Timestamped notes *(optional)* | A `[MM:SS]` timeline; off by default |
| **Chatbot** | Ask anything about the recording. Answers come from the transcript, with timestamps (clickable for YouTube videos) |

More features:

- **YouTube links:** the app uses the video's captions. If there are none, it downloads the audio and transcribes it.
- **Uploaded files:** MP3, WAV, M4A, MP4, MKV, WEBM and more are transcribed with speech-to-text.
- **Speech-to-text engines:** *Whisper* (faster-whisper, runs locally and free) or *Gemini* (cloud).
- **A critic agent** checks every output against the transcript and sends flawed drafts back to the agent that wrote them.
- **Output languages:** English, Hindi, Telugu, Spanish, French and German, handled by a translator agent.
- **Export** as Markdown, text or PDF, plus the mind map as a web page and the full transcript.

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

Open the URL Streamlit prints (usually http://localhost:8501), paste a YouTube link or upload a file, and click **Generate notes**. The live log shows each agent as it works. When the notes are ready, the results appear in tabs, and the **💬 Ask about this recording** box below them lets you ask questions.

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
 │       ┌─────────────┬──────────┼──────────┬─────────────┐ (in parallel)    │
 │       ▼             ▼          ▼          ▼             ▼                   │
 │  short_summary long_summary mind_map  takeaways  (timeline, optional)      │
 │       └─────────────┴──────────┼──────────┴─────────────┘                  │
 │                                ▼                                           │
 │                             critic ──(issues, < MAX_REVISIONS)──► revise ─┐│
 │                                │  ▲                                       ││
 │                                │  └───────────────────────────────────────┘│
 │                    (not English)│(English)                                  │
 │                  translate ◄───┴───► export ──► END                        │
 └──────────────────────────────────────────────────────────────────────────┘
                                       │ notes + transcript
                                       ▼
                    ┌──────────────────────────────────────┐
                    │  Chat agent: question + chat history │
                    │  → answer with [MM:SS] timestamps     │
                    └──────────────────────────────────────┘
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

### Mind map

The Mind Map agent writes a nested Markdown outline (central topic → 3–7 main branches → sub-points). An outline is easy for the model to write, for the critic to check, and for the translator to translate without breaking the structure. [tools/mindmap_tools.py](tools/mindmap_tools.py) parses it into a tree and renders it three ways:

- **In the app and as a download:** an interactive [markmap](https://markmap.js.org/) page. You can zoom, pan, and click a node to fold it. The page loads markmap 0.18.12 from the jsDelivr CDN, so it needs an internet connection.
- **In the Markdown notes:** a Mermaid `mindmap` code block, which renders on GitHub, in Obsidian and in VS Code.
- **In the PDF:** a tree diagram drawn with reportlab. No extra software is needed.

The critic checks the outline's structure in code: one central topic, 2–9 branches, at most 4 levels, and short labels. A malformed mind map is sent back to the Mind Map agent.

### Chatbot

The Chat agent ([agents/chat_agent.py](agents/chat_agent.py)) runs after the notes are made, once per question:

- **Normal recordings** (transcript up to `CHAT_FULL_CONTEXT_MAX_CHARS`, about 5 hours of speech): the whole transcript goes to Gemini. That's **one call per question**, and the model can answer questions about any part of the recording.
- **Very long recordings:** a local BM25 keyword search ([tools/retrieval.py](tools/retrieval.py)) picks the ~2-minute passages that best match the question. This search makes no API calls, and only those passages are sent.
- The short summary, key takeaways and mind map are always included for the big picture. The last few questions and answers are sent as conversation memory, so follow-up questions work.
- The model is told to answer only from the recording, cite `[MM:SS]` timestamps, say plainly when the recording doesn't cover something, reply in the language of the question, and ignore any instructions inside the transcript.

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
| `short_summary` | Short Summary agent (LLM) | 80–120-word summary | [agents/short_summary_agent.py](agents/short_summary_agent.py) |
| `long_summary` | Long Summary agent (LLM) | Detailed summary with a heading per topic | [agents/long_summary_agent.py](agents/long_summary_agent.py) |
| `mind_map` | Mind Map agent (LLM) | Nested outline for the mind map | [agents/mind_map_agent.py](agents/mind_map_agent.py) |
| `takeaways` | Takeaways agent (LLM) | 5–8 key takeaways | [agents/takeaways_agent.py](agents/takeaways_agent.py) |
| `timeline` | Timeline agent (LLM) | Chronological `[MM:SS]` notes (optional) | [agents/timeline_agent.py](agents/timeline_agent.py) |
| `critic` | Critic agent (LLM + code) | An LLM fact-check with structured output. Plus code checks: timeline timestamps must exist in the source, be in order and fall within the recording's length; the mind map outline must be well formed | [agents/critic_agent.py](agents/critic_agent.py) |
| `revise` | the writers again | Sends each failed output, with the critic's issues, back to the agent that wrote it | same |
| `translate` | Translator agent (LLM) | Translates the final outputs and keeps the Markdown and timestamps | [agents/translator_agent.py](agents/translator_agent.py) |
| `export` | Tool | Builds the final Markdown document | [agents/export_agent.py](agents/export_agent.py) |
| *(after the graph)* | Chat agent (LLM) | Answers questions about the recording | [agents/chat_agent.py](agents/chat_agent.py) |

The writer agents share [agents/analysis_base.py](agents/analysis_base.py), and [agents/registry.py](agents/registry.py) lists them in display order. Each can write a first draft and revise its own draft from feedback. They always write in English and leave translation to the translator agent, so the critic always compares English notes with the source.

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
│   ├── mindmap_tools.py    # outline parsing; markmap, Mermaid and PDF rendering
│   ├── retrieval.py        # local BM25 transcript search for the chatbot
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
| `CHAT_FULL_CONTEXT_MAX_CHARS` | `300000` | Transcripts up to this size are given to the chatbot in full; longer ones are searched |
| `CHAT_SEARCH_PASSAGES` | `8` | Passages sent per question in search mode |
| `CHAT_HISTORY_TURNS` | `6` | Earlier question/answer pairs sent as conversation memory |
| `AVS_CACHE_DIR` | `.cache` | Where transcripts are cached |

The sidebar in the app sets the per-run options: which outputs to make, the language, the speech-to-text engine and Whisper model, the YouTube speech-to-text options, and whether the critic runs.

**Which Whisper model?** On a CPU, `base` takes a few seconds per minute of audio and is fine for clear speech. Use `small` for better accuracy, especially for Indian languages or accented speech. `medium` is best but slow without a GPU.

---

## Command-line use

```bash
python run_cli.py "https://youtu.be/VIDEO_ID"                       # notes to stdout
python run_cli.py lecture.mp3 -o notes.md                            # + notes_mindmap.html
python run_cli.py lecture.mp3 -o notes.md --chat                     # then ask questions
python run_cli.py "https://youtu.be/VIDEO_ID" --force-stt --stt-engine gemini
python run_cli.py talk.mp4 --language Hindi --outputs short_summary,mind_map,timeline
python run_cli.py --graph                                            # Mermaid diagram
```

Output names for `--outputs`: `short_summary`, `long_summary`, `mind_map`, `takeaways`, `timeline`.

Run `python run_cli.py -h` for all options.

---

## Tests

```bash
python -m pytest
```

The tests make **no API calls**:

- [tests/test_tools.py](tests/test_tools.py): link parsing, timestamps, chunking, the critic's timestamp check, audio split/merge (uses a real ffmpeg file).
- [tests/test_graph.py](tests/test_graph.py): the whole agent graph with a fake LLM. It covers the default outputs with a critic revision, a malformed mind map caught by the code check, the revision limit, the caption → speech-to-text fallback and the transcript cache, the fallback switched off, private videos, and map-reduce with translation.
- [tests/test_mindmap_and_chat.py](tests/test_mindmap_and_chat.py): outline parsing and checks, Mermaid/HTML escaping, clickable timestamps, BM25 search, and the chat agent's full-context and search modes and memory.
- [tests/test_llm_client.py](tests/test_llm_client.py): rate-limit waiting, model fallback, daily-quota and unknown-model errors.
- [tests/test_app.py](tests/test_app.py): the Streamlit UI (Streamlit `AppTest`): results tabs, chat with memory, clearing the chat, chat errors, and PDF export with the mind map.

---

## Gemini quotas

On the free tier, Gemini limits both **requests per minute** and **requests per day** for each model.

- A short video with captions needs about **5 calls**: 4 writers + 1 critic, plus 4 translations for other languages and any revisions.
- **Each chat question is 1 call.**
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
- **The interactive mind map** loads markmap from a CDN, so it needs internet access. The PDF and Mermaid versions work offline.
- **Chat in search mode** (very long recordings only) uses keyword matching. A question worded very differently from the speaker, or in another language than the transcript, may miss the right passage.
- **Chat history** is kept for the browser session only. It isn't saved when the app restarts.

---

## Ideas for extending it

- **Semantic search** for the chatbot on very long recordings (embeddings instead of keywords).
- **Saved chats and notes** with LangGraph checkpointing (`langgraph-checkpoint-sqlite` is installed).
- **Speaker labels** (diarization) for interviews and meetings.
- **Local LLM:** replace `get_llm` with `ChatOllama` to run completely offline.
- **LangGraph checkpointing** (`langgraph-checkpoint-sqlite` is installed) to resume long runs and keep a history of past notes.
- **Tracing** with LangSmith to inspect every agent's prompt and answer.

---

## Security

Never commit `.env`; it contains your API key. The included `.gitignore` already excludes it, along with the transcript cache.
