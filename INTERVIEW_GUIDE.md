# AV Summarizer AI: Interview Preparation Guide

This guide explains the project end to end, how it is built, why it is built that way, and the questions an interviewer is likely to ask, with answers. Every answer is based on the actual code in this repository, so you can point to files if asked.

**How to use it:** read Part 1 to Part 4 once to understand the project, practise the pitch in Part 1 out loud, then go through the questions in Part 6. The cheat sheet in Part 7 has the numbers worth memorising.

---

## Contents

1. [The pitch (what to say first)](#part-1-the-pitch)
2. [Problem, users and features](#part-2-problem-users-and-features)
3. [Tech stack and why each piece was chosen](#part-3-tech-stack-and-why)
4. [Architecture](#part-4-architecture)
    - 4.1 Layers
    - 4.2 The agent graph
    - 4.3 One request, start to finish
    - 4.4 Component deep dive
5. [Design decisions, trade-offs and limitations](#part-5-design-decisions-trade-offs-and-limitations)
6. [Interview questions and answers](#part-6-interview-questions-and-answers)
7. [Cheat sheet](#part-7-cheat-sheet)

---

## Part 1: The pitch

### 30-second version

> "I built AV Summarizer AI, a multi-agent application that turns YouTube videos and audio or video files into study notes: a summary, key takeaways and timestamped notes. It's built with LangGraph for orchestration and LangChain with Google Gemini as the LLM. For YouTube it uses captions, and when there are none it downloads the audio and transcribes it with Whisper running locally. A critic agent fact-checks every output against the transcript and sends bad drafts back for revision. It supports six output languages, exports to Markdown and PDF, and has both a Streamlit web UI and a CLI."

### 2-minute version (add these after the 30 seconds)

> "Architecturally it's a LangGraph state graph. One shared state object flows through about a dozen nodes. I separated *agents*, nodes where an LLM has to think, like writing, judging or translating, from *tools*, deterministic steps like downloading audio, converting it with ffmpeg, chunking text and exporting.
>
> A few things I'm proud of:
> - **Long recordings** are handled with map-reduce: the transcript is chunked on timestamp boundaries, notes are written for each chunk in parallel, and merged until they fit in one request.
> - **The three writer agents run in parallel**, then the critic reviews them. The critic combines an LLM review with structured Pydantic output and a code-based check that every timestamp really exists in the source. Failed outputs go back to the agent that wrote them, at most twice.
> - **Resilience:** every Gemini call goes through one client module that waits out rate limits, retries with exponential backoff, and when a model is overloaded (HTTP 503) automatically falls back to another model.
> - **Caching:** transcripts are cached on disk by video ID or file hash, so the same audio is never transcribed twice.
>
> It has 43 pytest tests, including full graph runs with a fake LLM, so the tests don't need an API key."

---

## Part 2: Problem, users and features

### The problem
Students and professionals watch long lectures, talks and meetings. Re-watching to find one point is slow. Existing summarizers often:
- fail when a video has no captions,
- **hallucinate** facts or timestamps,
- can't handle multi-hour recordings because of the LLM's context window,
- only work in English.

### Who it's for
Students revising lectures, people catching up on meetings or podcasts, anyone who wants notes in their own language.

### Features

| Feature | How it works |
|---|---|
| YouTube input | Captions via `youtube-transcript-api`; if there are none, audio is downloaded with `yt-dlp` and transcribed |
| File upload | MP3, WAV, M4A, MP4, MKV, WEBM, MOV and more (13 extensions), up to 1 GB |
| Two speech-to-text engines | **Whisper** (faster-whisper, local, free, accurate timestamps) or **Gemini** (cloud, no local model) |
| Three outputs | Summary (Short / Medium / Detailed), 5–8 key takeaways, timestamped notes |
| Quality control | Critic agent: LLM fact-check + code timestamp validation + revision loop |
| Long recordings | Map-reduce over timestamped chunks |
| Six languages | English, Hindi, Telugu, Spanish, French, German (translator agent) |
| Export | Markdown, text, PDF, and the full transcript |
| Two interfaces | Streamlit web UI with live progress, and a CLI (`run_cli.py`) |
| Resilience | Retries, rate-limit waits, model fallback on overload |

---

## Part 3: Tech stack and why

| Technology | Role | Why this choice |
|---|---|---|
| **Python 3.10+** | Language | Best ecosystem for AI, audio and data |
| **LangGraph** | Orchestrates the agents as a state graph | Supports conditional routing, **parallel fan-out**, **loops** (critic → revise → critic) and streaming. A simple chain can't loop or branch cleanly |
| **LangChain** (`langchain-core`, `langchain-google-genai`) | Prompt templates, chat model wrapper, structured output | `ChatPromptTemplate` for reusable prompts; `with_structured_output()` turns answers into Pydantic objects; swapping provider means changing one function |
| **Google Gemini** (default `gemini-3.5-flash`) | The LLM | Large context window, multimodal (can take audio directly), free tier for development |
| **faster-whisper** | Local speech-to-text | A CTranslate2 re-implementation of OpenAI Whisper; up to ~4× faster with int8 on CPU; free and private |
| **youtube-transcript-api** | Fetches captions | No API key needed, fast, gives timestamps |
| **yt-dlp** | Downloads YouTube audio | The most maintained YouTube downloader |
| **ffmpeg / ffprobe** | Converts audio to 16 kHz mono WAV, cuts pieces, reads duration | Industry standard; handles any container or codec |
| **Pydantic** | Schemas for structured LLM output | Validated, typed objects instead of parsing free text |
| **Streamlit** | Web UI | Fast to build a data/AI UI in pure Python |
| **ReportLab** | PDF export | Pure-Python PDF generation |
| **python-dotenv** | Loads `.env` | Keeps the API key out of the code |
| **pytest** | Tests | Fixtures and `monkeypatch` make faking the LLM easy |

---

## Part 4: Architecture

### 4.1 Layers

```
┌──────────────────────────────────────────────────────────────┐
│ PRESENTATION      app.py (Streamlit)        run_cli.py (CLI) │
├──────────────────────────────────────────────────────────────┤
│ ORCHESTRATION     graph/builder.py   builds + compiles graph │
│                   graph/router.py    routing decisions       │
│                   graph/state.py     shared PipelineState    │
│                   graph/runner.py    stream → simple events  │
├──────────────────────────────────────────────────────────────┤
│ AGENTS (LLM)      summary, takeaways, timeline, critic,      │
│                   chunk_notes, translator                    │
│ NODES (no LLM)    ingest, transcription, preprocess, export  │
├──────────────────────────────────────────────────────────────┤
│ TOOLS             youtube_tools, audio_tools, stt_tools,     │
│                   text_tools, export_tools, cache            │
├──────────────────────────────────────────────────────────────┤
│ LLM GATEWAY       llm/client.py — the ONLY code that calls   │
│                   Gemini: retries, rate limits, fallback     │
├──────────────────────────────────────────────────────────────┤
│ CROSS-CUTTING     config.py (settings from .env)             │
│                   errors.py (exception hierarchy)            │
└──────────────────────────────────────────────────────────────┘
```

**Key rule:** each layer only calls the layer below it. The UI never calls Gemini directly. Agents never call `ChatGoogleGenerativeAI` directly; they all go through `llm/client.py`.

### 4.2 The agent graph

```
START
  │
  ▼
ingest ──(upload)──────────────────────────────────────┐
  │ (youtube)            (force_stt)                   │
  ├────────────────────────────┐                       │
  ▼                            ▼                       ▼
fetch_captions ──(no captions)──► download_audio ──► speech_to_text
  │ (captions ok)                 │ (cached transcript)    │
  └──────────────┬───────────────┘◄──────────────────────┘
                 ▼
            preprocess ──(long: > 60,000 chars)──► chunk_notes (map-reduce)
                 │ (short)                               │
                 └──────────── fan-out (parallel) ───────┘
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
           summary       takeaways      timeline
              └──────────────┼──────────────┘
                             ▼
                          critic ──(issues & revisions < 2)──► revise ──┐
                             │  ▲                                       │
                             │  └───────────────────────────────────────┘
               (not English) │ (English)
              translate ◄────┴────► export ──► END
                  └──────────────────►┘
```

You can print the real compiled graph: `python run_cli.py --graph` (outputs a Mermaid diagram).

**Node types:**

| Node | LLM? | File |
|---|---|---|
| `ingest` | No | `agents/ingestion_agent.py` |
| `fetch_captions`, `download_audio`, `speech_to_text` | Only if Gemini STT is chosen | `agents/transcription_agent.py` |
| `preprocess` | No | `agents/preprocess.py` |
| `chunk_notes` | Yes (map + reduce) | `agents/chunk_notes_agent.py` |
| `summary`, `takeaways`, `timeline` | Yes | `agents/*_agent.py` + `agents/analysis_base.py` |
| `critic` | Yes + code checks | `agents/critic_agent.py` |
| `revise` | Yes (reuses the writer agents) | `agents/critic_agent.py` |
| `translate` | Yes | `agents/translator_agent.py` |
| `export` | No | `agents/export_agent.py` |

### 4.3 One request, start to finish

Example: a user pastes a 20-minute YouTube lecture link, chooses Medium length, all three outputs, Hindi.

1. **UI** (`app.py`): `extract_video_id()` validates the link immediately and shows the thumbnail. On click, it builds the initial state: `{"source_type": "youtube", "url": ..., "options": {...}}` and calls `stream_pipeline(state)`.
2. **ingest**: extracts the 11-character video ID and sets `source_key = "yt_<id>"` (used for caching).
3. **Router `after_ingest`**: YouTube and not forced to STT → `fetch_captions`.
4. **fetch_captions**: downloads captions (prefers `en`, `en-US`, `en-GB`, `hi`; otherwise any language). Each library error is mapped to a `TranscriptError` with a `reason`.
   - If captions are **disabled / missing / empty / blocked**, speech-to-text can help → it records `caption_problem` and the router sends it to `download_audio`.
   - If the video is **private / deleted / age-restricted**, STT can't help → it stops with a clear error.
5. **preprocess** (no LLM): cleans text (HTML entities, `[Music]` tags), merges tiny caption fragments into ~30-second lines tagged `[MM:SS]`, splits into chunks of ≤ 24,000 characters, and computes total length. 20 minutes ≈ 15,000 characters < 60,000 → **single pass**.
6. **Router `after_preprocess`** returns a **list** `["summary", "takeaways", "timeline"]` → LangGraph runs all three **in parallel**.
7. **Writer agents**: each fills a shared prompt template (role, source kind, duration, its own instruction, source text) and calls `client.ask()`. Always in English.
8. **critic**: one structured LLM call reviews all three (`Critique` Pydantic schema), and `validate_timestamps()` checks the timeline in code. Say the timeline has a timestamp `[25:10]` in a 20-minute video → flagged.
9. **Router `after_critic`**: timeline failed, `revision_count` 0 < 2 → `revise`. Only the timeline agent rewrites its draft with the issues list. Back to **critic**, which only re-reviews the failed output. Passes.
10. **Router**: language is Hindi → **translate**: all three outputs are translated in parallel (`ask_many`), keeping Markdown and `[MM:SS]` timestamps.
11. **export**: builds the final Markdown document with title, source link and a note about where the transcript came from.
12. **UI**: shows results in tabs, the agent trace, the critic's remaining concerns (if any), and download buttons. PDF is not offered for Hindi/Telugu (ReportLab can't shape those scripts).

Throughout, nodes call `emit()` to send live progress, which the UI shows as "⏳ Summary agent: writing…" with a progress bar.

### 4.4 Component deep dive

#### Shared state — `graph/state.py`
- `PipelineState` is a `TypedDict` with `total=False`, grouped by stage: input, ingestion, transcription, preprocessing, analysis, quality control, output.
- Every node receives the full state and **returns only the keys it changes**; LangGraph merges them.
- `log: Annotated[List[str], operator.add]` is a **reducer**. Without it, when three agents run in parallel and each returns `log`, LangGraph would raise a conflict error (multiple writes to the same key in one step). With `operator.add`, the lists are concatenated.

#### Orchestrator — `graph/router.py`
Plain functions that look at the state and return the next node name, or a **list of names for parallel fan-out**.
- `after_ingest`: upload → STT; YouTube + `force_stt` → download; else → captions.
- `after_captions` / `after_download`: if `segments` exist → preprocess.
- `after_preprocess`: long → `chunk_notes`; else the list of selected writers.
- `after_critic`: failures and `revision_count < MAX_REVISIONS (2)` → `revise`; else translate or export.

These are **rule-based on purpose**: the decisions have one correct answer, and rules are free, instant, deterministic and testable. LLMs are used only where judgement or language is needed.

#### Graph builder — `graph/builder.py`
- `StateGraph(PipelineState)`, `add_node`, `add_edge`, `add_conditional_edges`, then `compile()`.
- Edges from each writer to `critic` mean the critic runs **once**, after all writers in the same step finish (LangGraph executes in "super-steps").
- `get_graph()` uses `@lru_cache(maxsize=1)` so the graph is compiled once per process.

#### Runner — `graph/runner.py`
`graph.stream(state, stream_mode=["custom", "updates", "values"])`:
- `custom` → progress messages sent by `emit()` (via `get_stream_writer()`),
- `updates` → "node X finished" with its log lines,
- `values` → the full state after each step; the last one is the final result.

It turns these into three simple event types (`progress`, `step`, `done`), so the **UI and CLI don't depend on LangGraph internals**.

#### Writer agents — `agents/analysis_base.py`
- One class `AnalysisAgent(key, role, instruction)` with two methods: `node()` (first draft) and `revise()` (rewrite using the critic's issues).
- Summary, Takeaways and Timeline agents are each ~10 lines: they only provide a role name and an instruction function. This is the **Template Method / Strategy** idea: shared behaviour, a pluggable instruction.
- `agents/registry.py` maps output keys to agents. Adding a new output type (e.g. "quiz") = a new agent file + one registry line.
- Prompt guardrails: *"Use ONLY the information in the SOURCE; never invent facts, names, numbers or timestamps."*

#### Map-reduce — `agents/chunk_notes_agent.py`
- **Map**: one prompt per chunk, run in parallel with `ask_many` (concurrency 4), each producing bullet notes starting with `[MM:SS]`.
- **Reduce**: while the combined notes exceed 40,000 characters, group the notes into groups of ~20,000 characters and merge each group. Repeat until it fits, or stop if a pass can't shrink the number of parts (prevents an infinite loop).
- The writers then work on the condensed notes (`source_kind = "condensed notes"`) instead of the raw transcript.

#### Critic — `agents/critic_agent.py`
Two independent checks:
1. **LLM review** with structured output (`Critique` → list of `OutputReview{output, passed, issues}`): unsupported claims, timestamps not in the source, missing major topics, ignoring the task.
2. **Code check** `validate_timestamps()` for the timeline:
   - no timestamps at all → issue,
   - timestamps after the end of the recording (+5 s) → issue,
   - timestamps more than 60 s from any timestamp in the source (binary search with `bisect_left`) → issue,
   - not in chronological order → issue.

Why both? LLMs are bad at precise numeric checking; code is perfect at it. LLMs are good at judging meaning; code can't do that.

Details worth knowing:
- Only outputs that **failed** are re-reviewed in later rounds (saves calls).
- If the LLM review itself fails, the run continues with only the code checks (**graceful degradation**).
- `revise` rewrites the failed outputs **in parallel** with a `ThreadPoolExecutor`.
- The loop is capped by `MAX_REVISIONS = 2`. If problems remain, the UI shows a warning with the critic's remaining concerns instead of hiding them.
- The writers always write English, so the critic always compares English notes with the source. Translation happens **after** review.

#### Transcription — `agents/transcription_agent.py`, `tools/stt_tools.py`, `tools/audio_tools.py`
- All sources are normalised to the **same segment format**: `[{"text", "start", "duration"}]`. Captions, Whisper and Gemini all produce it, so everything downstream is source-independent.
- Audio is first converted by ffmpeg to **16 kHz mono WAV** (what Whisper expects).
- **Whisper**: `faster-whisper` uses the GPU (float16) if CUDA is available, else CPU (int8). Uses VAD (voice-activity detection) to skip silence and beam size 5. The model is loaded once (`lru_cache`). The segment generator lets us report progress while transcribing.
- **Gemini STT**: the audio is cut into 10-minute MP3 pieces with 10 s overlap. Each piece is sent with a structured schema (`AudioTranscript`). Timestamps are shifted to absolute time; overlap duplicates are removed by giving each piece ownership of the segments that start between the **midpoints of its overlaps**, so nothing is lost or duplicated.
- **Cache** (`tools/cache.py`): JSON files in `.cache/transcripts/`. Key = `yt_<videoId>` or `file_<sha1 of content>` + engine + Whisper model size. Best-effort: cache failures never break the run.
- Temporary folders are always removed in a `finally` block.

#### LLM gateway — `llm/client.py`
The **only** module that talks to Gemini. Three public functions:
- `ask(prompt)` → text,
- `ask_many(prompts)` → parallel via LangChain's `batch()` with `max_concurrency`; failed items are retried one by one through `ask`,
- `ask_structured(schema, prompt)` → a validated Pydantic object.

Error handling strategy:

| Error | Detection | Action |
|---|---|---|
| Invalid API key | `API_KEY_INVALID`, `PERMISSION_DENIED` | Fail immediately with a clear message (retrying can't help) |
| Per-minute rate limit (429, short wait) | Parse Google's "retry in N s" | Sleep N+1 s and retry (up to 4 times, max 90 s each). Doesn't count as a failed attempt |
| Daily quota (429, long wait) | Wait > 90 s | Switch to the next model |
| Model overloaded (503) | `UNAVAILABLE`, `503`, "high demand" | Retry once, then switch to the next model |
| Model not found (404) | `NOT_FOUND` | Switch to the next model |
| Network / empty answer / other | — | Exponential backoff: 3 s, 6 s, 12 s… up to 3 attempts |

**Model fallback chain:** `gemini-3.5-flash` → `gemini-2.5-flash` → `gemini-3.1-flash-lite` → `gemini-flash-latest` → `gemini-2.5-flash-lite` (configurable with `GEMINI_FALLBACK_MODELS`).
- The model that worked is remembered for 10 minutes ("sticky"), so later calls don't keep hitting an overloaded model. A `threading.Lock` protects this shared state.
- If every model is busy, it waits 20 s and tries the whole list again (3 rounds) before giving up.
- `get_llm(temperature, model)` is `lru_cache`d, so client objects are reused.
- LangChain's own retries are turned off (`max_retries=0`) so our code can read Google's wait time and decide.

**Temperatures:** writers 0.3 (a little fluency), chunk notes 0.2, translation 0.1, critic and structured output 0.0 (deterministic judgement).

#### Errors — `errors.py`
```
PipelineError                 (base: stops the run with a user-friendly message)
├── InvalidInput
│   └── InvalidYouTubeURL
├── TranscriptError(reason)   (.stt_can_help: is speech-to-text worth trying?)
├── AudioError
└── LLMError
    └── ModelUnavailable(transient)   (in llm/client.py)
```
The UI catches `PipelineError` and shows `str(e)` directly, so every message is written for end users. Anything else is caught as a last resort and shown with an expandable traceback, so the app never crashes to a blank page.

#### UI — `app.py`
- Sidebar: length, outputs, language, STT engine and Whisper size, caption fallback options, critic toggle.
- Validates the YouTube link as you type and shows the thumbnail.
- Uploaded files are written to a temp folder with a sanitised filename and always deleted afterwards.
- Live progress: `st.status`, a progress bar with rough per-node weights, and a line per finished step.
- Results are stored in `st.session_state` with an **identity** of the input, so old results aren't shown after the input changes (Streamlit re-runs the whole script on every interaction).

#### Tests — `tests/` (43 tests, ~3 seconds)
| File | What it covers |
|---|---|
| `test_tools.py` | URL parsing (many URL shapes + rejects), time helpers, text cleaning, chunking covers everything within the limit, timestamp validation, overlap merging, real ffmpeg split |
| `test_graph.py` | **Whole graph runs with a fake LLM**: captions + one revision, revisions stop at max, missing captions → STT, fallback disabled → error, private video doesn't fall back, long upload → map-reduce + translation |
| `test_llm_client.py` | Retry-delay parsing, rate-limit waits, daily quota, unknown model, backoff, and the model fallback chain (overload switch, sticky model, all busy, invalid key doesn't fall back). `time.sleep` is monkeypatched so tests are instant |
| `test_app.py` | Streamlit `AppTest`: initial page, invalid link error, full run shows results, pipeline error shown, PDF export |

---

## Part 5: Design decisions, trade-offs and limitations

### Key decisions

| Decision | Why | Trade-off |
|---|---|---|
| LangGraph instead of a simple chain | Needed branching, parallelism and a loop | More concepts to learn |
| Rule-based routing, not an LLM "supervisor" | Decisions have clear right answers; rules are free, fast, testable | Less flexible for unexpected inputs |
| Agents vs tools separation | LLMs only where judgement is needed | — |
| Writers always in English, translate last | Critic compares like with like; one translation step | An extra LLM call per output for non-English |
| Critic = LLM + code | Each covers the other's weakness | More calls; capped at 2 revisions |
| One LLM gateway | Retries, rate limits, fallback in one place; provider swap in one function | — |
| Same segment format for every source | Downstream code doesn't care where text came from | — |
| Chunk on timestamp lines, not raw characters | Never cuts a sentence or loses a timestamp | Chunks are approximately, not exactly, the max size |
| Local Whisper as default STT | Free, private, accurate timestamps | Slow on CPU for large models |
| Disk cache by content hash | Never pay twice to transcribe the same audio | No eviction policy yet |

### Known limitations (be honest about these if asked)
- **Accuracy depends on the transcript.** The critic checks notes against the transcript, not the audio. Bad auto-captions → bad notes.
- **Gemini STT timestamps** are approximate (a few seconds).
- **YouTube may block** caption or audio downloads from cloud/VPN IPs.
- **Translations are not re-checked** by the critic.
- **Streamlit is single-process and synchronous**; a long run blocks that user session. Not built for many concurrent users.
- **The sticky-model state is in memory**, per process; it would not be shared across multiple server instances.
- **No authentication**, no persistence of past results, no cache eviction.

### How I would take it to production
1. **Split UI and backend**: a FastAPI service exposing `POST /jobs` and `GET /jobs/{id}`; the frontend polls or uses WebSockets/SSE for progress.
2. **Job queue**: Celery or RQ with Redis (or a cloud queue) so long jobs run in workers, scale horizontally and survive restarts.
3. **LangGraph checkpointing** (e.g. Postgres or SQLite checkpointer) to resume a failed run from the last completed node.
4. **Shared cache and storage**: Redis for transcript cache, S3/GCS for audio and results, with TTL eviction.
5. **Containers**: Docker image with ffmpeg; Kubernetes with separate GPU worker pools for Whisper.
6. **Observability**: LangSmith or OpenTelemetry tracing per agent; metrics for latency, token usage, cost, revision rate, critic pass rate.
7. **Security**: API key in a secret manager, authentication, rate limiting per user, file size and type limits (already partly there), virus scanning of uploads.
8. **CI/CD**: run pytest and linting on every PR (Jenkins/GitHub Actions), build and push the image, deploy to staging first.
9. **Evaluation**: a fixed set of videos with reference notes; track faithfulness and coverage scores over time when prompts or models change.

---

## Part 6: Interview questions and answers

> Tip: answer in this shape: **what** → **why** → **trade-off / what I'd improve**. Interviewers at service companies like EPAM value clear reasoning and awareness of trade-offs more than buzzwords.

### A. Project overview

**Q1. Tell me about your project.**
Use the 30-second pitch from Part 1, then stop and let them pick what to dig into.

**Q2. Why did you build it? What problem does it solve?**
Long lectures and meetings take a lot of time to re-watch. Existing tools fail without captions, hallucinate details, can't handle very long recordings, and are mostly English-only. My project handles all four: speech-to-text fallback, a critic agent for faithfulness, map-reduce for length, and a translator agent.

**Q3. What was your role?**
I designed and built it end to end: the agent graph, the prompts, the transcription pipeline, the LLM client, the UI, the CLI and the tests. *(Adjust this to be truthful about your own contribution.)*

**Q4. What is the most technically challenging part?**
Two good answers:
1. **Keeping outputs faithful.** LLMs invent timestamps. I solved it with a critic that combines an LLM review and a deterministic timestamp check, plus a bounded revision loop.
2. **Resilience against the API.** Rate limits, daily quotas and 503 overloads. I built one gateway that waits out per-minute limits using Google's own retry hint, uses exponential backoff for other errors, and falls back across a chain of models, remembering which one worked.

**Q5. What would you do differently if you started again?**
I would separate the backend (FastAPI + a job queue) from the UI from day one, add LangGraph checkpointing so long runs can resume, and set up an evaluation dataset early so prompt changes are measured, not judged by eye.

### B. Architecture

**Q6. Walk me through the architecture.**
Five layers: presentation (Streamlit and CLI), orchestration (LangGraph graph, router, shared state, runner), agents and nodes, tools (YouTube, audio, STT, text, export, cache), and one LLM gateway. Plus config and an error hierarchy used everywhere. Each layer only calls the one below it. Then describe the flow from Part 4.3.

**Q7. What is the difference between an "agent" and a "tool" in your project?**
An agent is a node where an LLM must use judgement or language: writing, critiquing, translating, condensing. A tool is a deterministic step with one correct answer: parsing a URL, downloading, converting audio, chunking text, building Markdown. I use LLMs only where they add value, because they're slower, cost money and are non-deterministic.

**Q8. Why LangGraph and not a plain LangChain chain or just Python functions?**
I needed three things a linear chain doesn't give cleanly: **conditional branching** (captions vs speech-to-text, single pass vs map-reduce), **parallel fan-out** (three writers at once) and a **cycle** (critic → revise → critic). LangGraph also gives shared typed state, reducers for parallel writes, and streaming of progress. With plain functions I would have had to build all of that myself.

**Q9. Is this really "agentic"? Your routing is just if-statements.**
Fair question. The *routing* is deliberately rule-based because those decisions have clear answers. The agentic parts are the **critic–revise loop**, where one agent evaluates another's work and the result decides the next step, and agents revising their own output from feedback. If needed I could add an LLM supervisor, but for this problem rules are cheaper, faster and more predictable. I'd rather use an LLM only where it is the best tool.

**Q10. How do agents communicate?**
Through the shared `PipelineState`, not by calling each other. Each node reads what it needs and returns only the keys it changed; LangGraph merges the updates. This keeps agents decoupled: the critic doesn't know how the summary was written, it just reads `state["summary"]`.

**Q11. How do the three writer agents run in parallel? How does the critic wait for all of them?**
The router returns a list of node names, so LangGraph schedules all of them in the same super-step and runs them concurrently. Each writer has an edge to `critic`, and LangGraph only moves to the next super-step after every node in the current one finishes, so the critic runs once, with all drafts available.

**Q12. What happens if two parallel nodes write to the same state key?**
LangGraph raises an error unless that key has a **reducer**. My `log` key is `Annotated[List[str], operator.add]`, so updates from parallel agents are concatenated. The writer outputs themselves use different keys (`summary`, `takeaways`, `timeline`), so they never conflict.

**Q13. How do you stop the critic–revise loop from running forever?**
`revision_count` in the state is incremented by every `revise`. The router only goes back to `revise` while there are failures **and** `revision_count < MAX_REVISIONS` (2). After that it moves on, and the UI clearly shows the critic's remaining concerns, so the user knows what to double-check.

**Q14. How does live progress reach the UI?**
Nodes call `emit()`, which uses LangGraph's `get_stream_writer()`. The runner streams the graph with `stream_mode=["custom", "updates", "values"]`: custom gives progress messages, updates gives "node finished" with its log, values gives the full state. The runner converts these to three simple event types, so the UI and CLI don't depend on LangGraph.

**Q15. How would you add a new output, e.g. a quiz generator?**
Create `agents/quiz_agent.py` with an `AnalysisAgent("quiz", "Quiz Agent", instruction)`, add one line to `agents/registry.py`, add a title in `config.OUTPUT_TITLES`, and add `"quiz"` to the router's allowed list. The graph builder creates nodes from the registry, so the parallel run, critic and revision work automatically. That's the open/closed principle in practice.

**Q16. Which design patterns did you use?**
- **Template Method / Strategy**: `AnalysisAgent` holds the shared draft/revise logic; each agent supplies its instruction.
- **Registry**: `ANALYSIS_AGENTS` maps keys to agents.
- **Gateway / Facade**: `llm/client.py` hides retries, rate limits and fallbacks behind `ask` / `ask_many` / `ask_structured`.
- **Adapter**: captions, Whisper and Gemini STT all produce the same segment format; `_fetch_raw` adapts two versions of the caption library.
- **Map-reduce** for long transcripts.
- **Retry with exponential backoff**, and a **fallback chain**.
- **Cache-aside** for transcripts.
- **Singleton-like caching** with `lru_cache` for the compiled graph, LLM clients and the Whisper model.

**Q17. How is the project structured, and why that way?**
By responsibility: `graph/` (orchestration), `agents/` (one file per agent), `tools/` (deterministic helpers), `llm/` (the gateway), plus `config.py`, `errors.py`, `app.py`, `run_cli.py` and `tests/`. Agents depend on tools and the gateway, never the other way around. That makes each part testable in isolation and easy to find.

### C. LLM and GenAI concepts

**Q18. How do you reduce hallucination?**
1. **Grounding prompts**: "Use ONLY the information in the SOURCE; never invent facts, names, numbers or timestamps."
2. **Low temperatures**: 0.0 for the critic, 0.1 for translation, 0.2–0.3 for writing.
3. **A critic agent** that fact-checks every output against the source.
4. **Deterministic checks in code** for timestamps (they must exist in the source, be in order, and be within the recording's length).
5. **A revision loop** that sends specific, actionable issues back to the author.
6. **Transparency**: if problems remain, the user is warned.

**Q19. What is temperature? Why different values?**
Temperature controls randomness when the model picks the next token. 0 is nearly deterministic; higher is more varied. The critic and structured outputs use 0.0 because judgement should be consistent. Translation uses 0.1 to be faithful. Writers use 0.3 for slightly more natural prose while staying grounded.

**Q20. What is structured output and why use it?**
Asking the LLM to return data matching a schema instead of free text. With LangChain's `with_structured_output(PydanticModel)`, Gemini returns JSON matching my `Critique` or `AudioTranscript` model, and I get a validated Python object. No fragile regex parsing, and if the structure is wrong, it's an error I can retry.

**Q21. What is a context window? How do you handle videos longer than it?**
The maximum number of tokens the model can read in one request. I measure the transcript in characters (~4 characters per token). Under 60,000 characters it's sent in one pass. Over that, I use **map-reduce**: chunk into ≤ 24,000-character pieces on timestamp line boundaries, write notes for each chunk in parallel (map), then merge groups of notes until they're under 40,000 characters (reduce, possibly several levels). The writers then work on the condensed notes.

**Q22. Why not just send the whole transcript to a model with a 1-million-token window?**
I could for many videos, but: cost and latency grow with input size; models attend less reliably to the middle of very long inputs ("lost in the middle"); and map-reduce works with any model, including smaller and cheaper ones. The threshold is configurable, so with a large-context model you can raise `SINGLE_PASS_MAX_CHARS`.

**Q23. Is this RAG? How is it different?**
No. RAG retrieves the most relevant pieces from a large knowledge base for a question. Here the task is to summarise **the whole** recording, so every part must be read; retrieval would skip content. Map-reduce is the right pattern for full coverage. RAG would fit a future "chat with the video" feature: embed the chunks, retrieve the relevant ones per question.

**Q24. How do you design your prompts?**
- A **system message** that sets the role ("You are the Timeline Agent…"), the grounding rule, the source type and duration, the language and format, and "no preamble".
- A **human message** with the specific TASK and the SOURCE, clearly separated.
- **Reusable templates** (`ChatPromptTemplate`) shared by all writers; only the instruction changes.
- The revise prompt includes the previous draft, the critic's issues as a list, and the source.
- Concrete format examples, e.g. `- **[MM:SS]** Topic: one-sentence description`.

**Q25. Why do the writers always write in English and then translate?**
So the critic compares English notes with the source consistently, and so prompts and quality are tuned once. Translation is a separate, well-defined task done at a low temperature, with instructions to keep Markdown and timestamps unchanged.

**Q26. How does the critic produce feedback the writer can act on?**
The schema requires `issues` to be "specific, actionable problems", and the prompt says each issue must state exactly what is wrong and how to fix it. Code issues are also specific, e.g. "These timestamps are later than the end of the recording (20:00): 25:10". The writer receives the list plus its previous draft.

**Q27. How would you evaluate output quality?**
Build a test set of videos with human-written reference notes. Measure **faithfulness** (are claims supported? LLM-as-judge or NLI models), **coverage** (are the main topics present?), **timestamp accuracy** (code check), and track the critic's pass rate and the number of revisions. Run it whenever prompts or models change.

**Q28. How would you reduce cost?**
Cache transcripts (done). Use rules instead of LLMs for routing (done). Re-review only failed outputs (done). Use a cheaper "lite" model for map and translation steps, cache LLM results for identical inputs, and let the user turn the critic off (done) for quick drafts.

### D. Resilience and error handling

**Q29. How do you handle Gemini API failures?**
All calls go through `llm/client.py`. Invalid key → fail immediately with a clear message. Per-minute rate limit → read Google's "retry in N seconds" and wait. Daily quota, model missing or 503 overload → switch to the next model in a fallback chain. Network errors or empty answers → exponential backoff (3 s, 6 s, 12 s). Every error becomes a readable `LLMError`.

**Q30. Tell me about a real problem you fixed.** *(Good STAR story)*
- **Situation:** users got "Gemini request failed after 3 attempts: model is overloaded (503)". It was Google's capacity problem, so the app was failing even though the code was correct.
- **Task:** make the app keep working when a model is overloaded.
- **Action:** I added a model fallback chain in the LLM gateway. On a 503 it retries once, then switches to the next model; on daily quota or missing model it switches immediately. The working model is remembered for 10 minutes so later calls don't waste time, protected by a lock. If all models are busy it waits 20 s and tries again, up to 3 rounds. Invalid API keys still fail fast, because switching can't help. I added 6 unit tests with a fake clock.
- **Result:** I tested it live while the main model was returning 503s. The first request switched models in about 16 seconds and the next one took under a second. All 43 tests pass.

**Q31. Why did you set `max_retries=0` in LangChain?**
LangChain's built-in retries don't know Google's suggested wait time and can't switch models. Turning them off lets my code decide: wait exactly as long as Google says for a rate limit, fail fast on a bad key, or move to another model on overload.

**Q32. What is exponential backoff and why use it?**
Waiting longer after each failure (3 s, 6 s, 12 s…). It gives a struggling service time to recover and avoids hammering it with retries, which would make the overload worse. In production I'd also add **jitter** (a random component) so many clients don't retry at the same moment.

**Q33. Why distinguish "retryable" and "non-retryable" errors?**
Retrying an invalid API key or a private video wastes time and gives the user a worse error later. That's why `TranscriptError` has a `reason` and a `stt_can_help` property: missing captions → try speech-to-text; private video → stop immediately with a clear message.

**Q34. What happens if the critic's LLM call itself fails?**
Graceful degradation: the critic logs a warning and applies only the code checks. Notes are still delivered. The critic is a quality add-on, so it shouldn't be a single point of failure.

**Q35. How do you make sure temporary files are cleaned up?**
Downloads and conversions go to `tempfile.mkdtemp()` folders that are removed in `finally` blocks, both in the speech-to-text node and in the UI for uploads. Uploaded filenames are also sanitised before being written.

### E. Audio and transcription

**Q36. Why convert audio to 16 kHz mono WAV?**
Whisper models are trained on 16 kHz mono audio. Converting first with ffmpeg also means any container or codec (MP4, MKV, M4A…) works, and video tracks are dropped (`-vn`).

**Q37. Whisper vs Gemini for speech-to-text — when would you use each?**
Whisper (local): free, private (audio never leaves the machine), precise timestamps, works offline, but slow on CPU for larger models. Gemini (cloud): no local model or GPU needed, fast for long audio, but costs API quota, needs internet, and its timestamps are approximate. Default is Whisper; I recommend it when timestamps matter.

**Q38. Why faster-whisper and not OpenAI's whisper package?**
faster-whisper uses CTranslate2, which is several times faster and uses less memory, with int8 quantisation on CPU. It also has built-in VAD (voice-activity detection) to skip silence. I use the GPU with float16 when CUDA is available and fall back to CPU automatically.

**Q39. How do you split long audio for Gemini without losing or duplicating words?**
Pieces of 10 minutes with 10 seconds of overlap. Each piece "owns" only the segments that start between the midpoints of its overlaps with its neighbours. After shifting timestamps to absolute time, I keep only owned segments and sort. A word in an overlap is kept exactly once. This is unit-tested.

**Q40. How does caching work?**
After speech-to-text, segments are saved as JSON in `.cache/transcripts/`. The key is the source (`yt_<videoId>` or `file_<SHA-1 of the file content>`) plus the engine and Whisper model size, because different engines give different transcripts. Before downloading or transcribing, the node checks the cache. Hashing the content (in 1 MB blocks to save memory) means a renamed file still hits the cache. Cache errors are ignored; it's best-effort.

**Q41. How do you handle all the different YouTube URL formats?**
`extract_video_id` accepts a bare 11-character ID, `watch?v=`, `youtu.be/`, `/shorts/`, `/embed/`, `/live/`, `m.` and `music.` hosts and `youtube-nocookie.com`. It checks the host against an allow-list and validates the ID with a regex. It's validated as the user types, so errors show before the run starts.

**Q42. Your caption code supports two library versions. Why?**
`youtube-transcript-api` changed its API between 0.x (static `get_transcript`) and 1.x (instance `fetch`), and some exception classes only exist in some versions. `_fetch_raw` detects the version with `hasattr`, and `_exc()` builds the exception tuple from the classes that exist. That keeps the app working across environments.

### F. Python and implementation details

**Q43. What does `@lru_cache` do here?**
It memoises function results. `get_graph()` compiles the graph once; `get_llm(temperature, model)` reuses client objects; `_load_whisper(size)` loads each model once, which matters because loading takes seconds and hundreds of MB.

**Q44. TypedDict vs Pydantic vs dataclass — where do you use each and why?**
- **TypedDict** for `PipelineState`: LangGraph works with dict state and partial updates; TypedDict gives type hints without runtime overhead.
- **Pydantic** for LLM structured output: needs runtime **validation** of untrusted model output.
- **dataclass** for simple internal records like `Chunk` and `AudioPiece`.

**Q45. How do you get parallelism in Python with the GIL?**
The work is **I/O-bound**: waiting for HTTP responses from Gemini. Threads release the GIL while waiting on network I/O, so `ThreadPoolExecutor` (in `revise`) and LangChain's `batch()` (in `ask_many`) give real concurrency. CPU-heavy Whisper runs in CTranslate2's native code, outside the GIL.

**Q46. Why is there a lock in the LLM client?**
The "last working model" is module-level state, and `ask` can be called from several threads at once (parallel revisions, Streamlit sessions). The `threading.Lock` makes reading and updating that state atomic.

**Q47. How is configuration handled?**
`config.py` loads `.env` with python-dotenv once and exposes typed settings with defaults (`_int()` falls back to the default if a value is invalid). Everything is overridable by environment variables, following the twelve-factor app principle. The API key is never in code, and `.env` is in `.gitignore`.

**Q48. Why use a generator in Whisper transcription?**
`model.transcribe()` returns a lazy generator; transcription actually happens while iterating. That lets me report progress after every segment ("Transcribed 04:10 of 20:00") instead of blocking silently until the end.

**Q49. How does Streamlit's execution model affect your code?**
Streamlit re-runs the whole script on every interaction. So results are stored in `st.session_state`, tagged with an identity of the input (video ID, or file name + size), and only shown if the identity still matches. That stops old results showing under a new link.

### G. Testing

**Q50. How did you test an app that depends on an LLM?**
I faked the LLM. In `test_graph.py`, a fixture monkeypatches `client.ask`, `ask_many` and `ask_structured` with a fake that returns scripted answers, then runs the **real compiled graph** end to end. That tests routing, parallelism, the revision loop, map-reduce and translation deterministically, without network or API key. The whole suite runs in ~3 seconds.

**Q51. What kinds of tests do you have?**
- **Unit tests** for pure functions: URL parsing, time helpers, chunking, timestamp validation, overlap merging.
- **Integration tests** of the full graph with a fake LLM, and one with real ffmpeg.
- **Resilience tests** for the LLM client: rate limits, quotas, overloads, fallbacks, with `time.sleep` monkeypatched so they're instant.
- **UI tests** with Streamlit's `AppTest`.

**Q52. Give an example of a test that catches a real bug.**
`test_revisions_stop_at_max`: the fake critic always fails the output, and the test checks the graph still finishes after exactly 2 revisions. Without the `MAX_REVISIONS` check that would be an infinite loop. Another: `test_unavailable_video_does_not_fall_back` makes sure we don't waste a download on a private video.

**Q53. What would you add to testing?**
Property-based tests for chunking (e.g. Hypothesis: every character ends up in exactly one chunk), contract tests against a recorded Gemini response, an evaluation suite for output quality, and running the tests in CI on every pull request.

### H. Scalability, deployment and system design

**Q54. How would you scale this to 10,000 users?**
- Split into a **FastAPI** backend and a separate frontend.
- Submit runs as **jobs** to a queue (Celery/RQ + Redis, or a cloud queue); return a job ID; stream progress over SSE/WebSocket.
- Run workers horizontally; a **separate GPU worker pool** for Whisper.
- Move the cache to **Redis** and files to **object storage** (S3/GCS) so all workers share them.
- Use **LangGraph checkpointing** in Postgres so a crashed job resumes.
- **Rate-limit per user** and manage a pool of API keys or a paid tier to stay under Gemini quotas.
- Add autoscaling based on queue length.

**Q55. What are the bottlenecks?**
1. Speech-to-text on CPU (minutes for long audio) → GPU workers or Gemini STT.
2. LLM rate limits and latency → concurrency limits, fallbacks, caching.
3. YouTube blocking downloads from cloud IPs → proxies or asking users to upload.
4. Streamlit being single-process → separate backend.

**Q56. How would you deploy it?**
A Dockerfile based on `python:3.12-slim` with ffmpeg installed and `pip install -r requirements.txt`; secrets via environment variables or a secret manager; health check on Streamlit's `/_stcore/health`; deploy on a VM, Cloud Run or Kubernetes. A CI pipeline (Jenkins or GitHub Actions) runs tests, builds the image, pushes it, and deploys to staging, then production.

**Q57. How would you monitor it in production?**
Tracing per agent (LangSmith or OpenTelemetry) to see each prompt, answer, latency and token count. Metrics: run duration per stage, error rate by type, fallback rate, critic pass rate, revisions per run, cost per run. Alerts on error spikes or quota exhaustion. Structured logs with a run ID.

**Q58. How would you make it work offline or with a different LLM provider?**
Change only `get_llm()` in `llm/client.py`, e.g. return `ChatOllama(model="llama3")`. All agents use the gateway, so nothing else changes. Whisper already runs locally.

**Q59. What about security and privacy?**
API key in `.env` (never committed) or a secret manager. Upload type and size limits, sanitised filenames, temp files deleted after use. Whisper keeps audio on the machine. For production: authentication, per-user rate limiting, encryption at rest for stored transcripts, a retention policy, and a note to users that Gemini STT sends audio to Google.

**Q60. How would you handle prompt injection, e.g. a video that says "ignore previous instructions"?**
The transcript is data, placed in a clearly delimited SOURCE section, and the system prompt defines the role. The output is only notes shown to the user; agents have no tools that take actions, so the impact is limited to bad notes. The critic also checks outputs against the source. For stronger protection: delimit the source more strictly, and check outputs for instruction-like content.

### I. Feature and product questions

**Q61. What are the limitations of your project?**
See Part 5: quality depends on the transcript; Gemini STT timestamps are approximate; YouTube can block cloud IPs; translations aren't re-checked; Streamlit isn't built for many concurrent users; the sticky-model state is per process; no authentication or history.

**Q62. What features would you add next?**
- **Chat with the video** (RAG over the transcript chunks).
- **Clickable timestamps** that open the video at that second (`&t=SECONDS`).
- **Speaker diarization** for meetings and interviews.
- **Quiz / flashcard** generation for students.
- **History of past runs** using LangGraph checkpointing.
- **Critic for translations.**

**Q63. Why Streamlit instead of React?**
Speed of development: a full interactive UI with uploads, progress and downloads in ~200 lines of Python. For a production product with many users, I'd move to a React or Angular frontend with a FastAPI backend.

**Q64. Why is PDF not available for Hindi and Telugu?**
ReportLab can't do complex text shaping for Indic scripts (combining characters render wrongly). Rather than produce a broken PDF, the app offers Markdown instead and says so. A fix would be generating HTML and converting it with a browser engine or WeasyPrint, which support shaping.

### J. Rapid-fire questions

| Question | Short answer |
|---|---|
| What's a node in LangGraph? | A function that takes the state and returns a partial update |
| What's a conditional edge? | A function that picks the next node(s) from the state |
| What's a reducer? | How LangGraph merges multiple updates to one key (`operator.add` for `log`) |
| What's a super-step? | One round where all scheduled nodes run, possibly in parallel |
| Max revisions? | 2 (`MAX_REVISIONS`) |
| Single-pass limit? | 60,000 characters (~15k tokens) |
| Chunk size? | 24,000 characters |
| Reduce target? | 40,000 characters |
| Parallel map calls? | 4 (`MAP_CONCURRENCY`) |
| Timestamp tolerance? | 60 seconds |
| Gemini STT piece size? | 10 minutes with 10 s overlap |
| Transcript line length? | ~30 seconds per `[MM:SS]` line |
| Default STT? | Whisper `base`, local |
| Cache key? | Source ID or file SHA-1 + engine + Whisper size |
| How many tests? | 43, ~3 seconds |
| How many languages? | 6 |
| Default model? | `gemini-3.5-flash`, with 4 fallbacks |

### K. Behavioural questions (EPAM asks these too)

**Q65. Describe a time you debugged something difficult.**
Use the 503 overload story (Q30) or the caption library version story (Q42): the library changed its API, so I made the code detect the version at runtime instead of pinning an old release.

**Q66. How do you make sure your code is maintainable?**
Single responsibility per file, one gateway for external calls, a typed shared state, user-friendly exceptions in one hierarchy, configuration in one place, docstrings explaining *why*, and tests for every routing branch.

**Q67. How do you learn a new technology like LangGraph?**
Official docs and examples first, then a tiny prototype (two nodes and a loop) before the real design, then tests to lock the behaviour down. The `legacy/` folder holds my first single-pipeline version, which I refactored into the agent graph once I understood the patterns.

**Q68. How would you explain this project to a non-technical person?**
"You give it a video link or a recording. A team of AI assistants listens to it, writes a summary, the key points and notes with times, and then a 'checker' assistant reads everything against what was actually said and makes the others fix mistakes. You can get the notes in your language and download them."

---

## Part 7: Cheat sheet

**One-line summary:** a LangGraph multi-agent pipeline that turns YouTube videos and audio/video files into grounded, timestamped study notes, with a critic–revise loop, map-reduce for long recordings, local Whisper fallback, six languages and a resilient Gemini gateway.

**Numbers:**
- ~13 nodes; 6 LLM agents (summary, takeaways, timeline, critic, chunk-notes, translator)
- 3 writers in parallel; critic + max 2 revisions
- 60k chars single pass · 24k chunks · 40k reduce target · 4 parallel calls
- 10-minute Gemini STT pieces, 10 s overlap · ~30 s transcript lines · 60 s timestamp tolerance
- Rate-limit wait ≤ 90 s, up to 4 times · backoff 3/6/12 s · 503: retry once then switch · all busy: 20 s × 3 rounds · sticky model 10 min
- 5 models in the chain · 6 languages · 13 file types · 1 GB uploads · 43 tests in ~3 s

**Files to remember:**
- `graph/builder.py`: the graph
- `graph/router.py`: routing decisions
- `graph/state.py`: shared state and the `log` reducer
- `agents/analysis_base.py`: shared writer logic and prompts
- `agents/critic_agent.py`: critic, timestamp validation, revise
- `agents/chunk_notes_agent.py`: map-reduce
- `llm/client.py`: retries, rate limits, model fallback
- `tools/audio_tools.py`: ffmpeg, split and merge with overlap
- `tests/test_graph.py`: full graph runs with a fake LLM

**Three things to say if you only get one chance:**
1. "LLMs only where judgement is needed; everything deterministic is plain code."
2. "A critic combining an LLM review and code checks, with a bounded revision loop, keeps outputs faithful."
3. "One LLM gateway handles rate limits, backoff and model fallback, so the app keeps working when the API doesn't."
