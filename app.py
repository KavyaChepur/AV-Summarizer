"""Streamlit UI for the agentic AV summarizer."""
import shutil
import tempfile
from pathlib import Path

import streamlit as st

import config  # loads .env
from errors import InvalidYouTubeURL, PipelineError
from graph.builder import NODE_LABELS
from graph.runner import stream_pipeline
from tools.audio_tools import AUDIO_EXTENSIONS
from tools.export_tools import PDF_UNSUPPORTED_LANGUAGES, markdown_to_pdf
from tools.stt_tools import WHISPER_SIZES
from tools.text_tools import fmt_time
from tools.youtube_tools import extract_video_id

st.set_page_config(page_title="AV Summarizer AI", page_icon="🎬", layout="centered")
st.title("🎬 AV Summarizer AI")
st.caption("A team of AI agents turns YouTube videos and audio/video files into notes.")

TITLE_TO_KEY = {v: k for k, v in config.OUTPUT_TITLES.items()}
AUDIO_PREVIEW = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".opus"}
# Rough share of the run each node represents, for the progress bar.
NODE_PROGRESS = {"ingest": 0.03, "fetch_captions": 0.08, "download_audio": 0.15,
                 "speech_to_text": 0.45, "preprocess": 0.5, "chunk_notes": 0.62,
                 "summary": 0.72, "takeaways": 0.72, "timeline": 0.72, "critic": 0.82,
                 "revise": 0.85, "translate": 0.93, "export": 1.0}
SOURCE_LABELS = {"captions": "YouTube captions", "whisper": "Whisper speech-to-text",
                 "gemini": "Gemini speech-to-text"}


def show_image(url: str) -> None:
    try:
        st.image(url, width="stretch")
    except TypeError:  # older Streamlit
        st.image(url, use_container_width=True)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Outputs")
    length = st.radio("Summary length", list(config.LENGTHS), index=1, horizontal=True)
    wanted = st.multiselect("Outputs", list(config.OUTPUT_TITLES.values()),
                            default=list(config.OUTPUT_TITLES.values()))
    language = st.selectbox("Output language", config.LANGUAGES)

    st.header("Transcription")
    engine_label = st.radio("Speech-to-text engine",
                            ["Whisper (local, free)", "Gemini (cloud)"],
                            index=0 if config.STT_ENGINE == "whisper" else 1,
                            help="Used for uploaded files, and for YouTube videos "
                                 "without captions.")
    stt_engine = "whisper" if engine_label.startswith("Whisper") else "gemini"
    whisper_model = config.WHISPER_MODEL
    if stt_engine == "whisper":
        whisper_model = st.select_slider(
            "Whisper model", WHISPER_SIZES,
            value=config.WHISPER_MODEL if config.WHISPER_MODEL in WHISPER_SIZES else "base",
            help="Bigger models are more accurate but slower. 'base' is a good start on "
                 "a CPU; a model is downloaded the first time it is used.")
    allow_fallback = st.checkbox("YouTube: use speech-to-text if there are no captions",
                                 value=True)
    force_stt = st.checkbox("YouTube: always use speech-to-text (ignore captions)",
                            value=False)

    st.header("Quality")
    use_critic = st.toggle("Critic agent reviews and fixes the notes", value=True)

    with st.expander("How it works"):
        st.markdown(
            "1. **Ingestion agent** checks the link or file.\n"
            "2. **Transcription agent** gets YouTube captions, or downloads the audio "
            "and runs speech-to-text.\n"
            "3. **Preprocessing** cleans the text and splits it into chunks.\n"
            "4. **Chunk-notes agent** condenses long recordings (map-reduce).\n"
            "5. **Summary, Takeaways and Timeline agents** write in parallel.\n"
            "6. **Critic agent** checks facts and timestamps and sends drafts back "
            f"for revision (up to {config.MAX_REVISIONS} times).\n"
            "7. **Translator agent** translates if needed, then the notes are exported.")

# ------------------------------------------------------------------ input
source = st.radio("Source", ["YouTube link", "Upload audio/video"], horizontal=True,
                  label_visibility="collapsed")

identity = None   # identifies the current input, so old results aren't shown for new input
video_id = None
uploaded = None

if source == "YouTube link":
    link = st.text_input("YouTube video link", placeholder="https://www.youtube.com/watch?v=…")
    if link:
        try:
            video_id = extract_video_id(link)
            identity = ("yt", video_id)
            show_image(f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg")
        except InvalidYouTubeURL as e:
            st.error(str(e))
else:
    uploaded = st.file_uploader("Audio or video file",
                                type=sorted(e.lstrip(".") for e in AUDIO_EXTENSIONS))
    if uploaded is not None:
        identity = ("file", uploaded.name, uploaded.size)
        if Path(uploaded.name).suffix.lower() in AUDIO_PREVIEW:
            st.audio(uploaded)

outputs = [TITLE_TO_KEY[w] for w in wanted]
run = st.button("Generate notes", type="primary",
                disabled=identity is None or not outputs)

# ------------------------------------------------------------------ run
if run:
    state = {"options": {"length": length, "outputs": outputs, "language": language,
                         "stt_engine": stt_engine, "whisper_model": whisper_model,
                         "force_stt": force_stt, "allow_stt_fallback": allow_fallback,
                         "use_critic": use_critic}}
    upload_dir = None
    if uploaded is not None:
        upload_dir = Path(tempfile.mkdtemp(prefix="avs_upload_"))
        safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in uploaded.name)
        path = upload_dir / safe_name
        path.write_bytes(uploaded.getbuffer())
        state.update(source_type="upload", file_path=str(path), file_name=uploaded.name)
    else:
        state.update(source_type="youtube", url=link)

    trace, final = [], None
    st.session_state.pop("result", None)
    try:
        with st.status("Agents are working…", expanded=True) as status:
            bar = st.progress(0.0)
            live = st.empty()
            done = 0.0
            for ev in stream_pipeline(state):
                if ev["type"] == "progress":
                    label = NODE_LABELS.get(ev["agent"], ev["agent"]).split(" · ")[0]
                    live.caption(f"⏳ **{label}:** {ev['message']}")
                    status.update(label=f"{label}: {ev['message']}")
                elif ev["type"] == "step":
                    done = max(done, NODE_PROGRESS.get(ev["node"], done))
                    bar.progress(done)
                    for line in ev["log"]:
                        entry = f"**{NODE_LABELS.get(ev['node'], ev['node'])}** — {line}"
                        trace.append(entry)
                        st.markdown(f"✅ {entry}")
                else:
                    final = ev["state"]
            live.empty()
            status.update(label="Done", state="complete", expanded=False)
    except PipelineError as e:
        st.error(str(e))
    except Exception as e:  # last-resort guard so the app never shows a raw traceback
        st.error(f"Unexpected error: {e}")
        with st.expander("Details"):
            st.exception(e)
    finally:
        if upload_dir:
            shutil.rmtree(upload_dir, ignore_errors=True)

    if final and final.get("markdown"):
        pdf = None
        if language not in PDF_UNSUPPORTED_LANGUAGES:
            try:
                pdf = markdown_to_pdf(final["markdown"])
            except Exception:
                pdf = None
        st.session_state["result"] = {
            "identity": identity, "trace": trace, "pdf": pdf, "language": language,
            **{k: final.get(k) for k in ("final_outputs", "markdown", "transcript_text",
                                         "reviews", "transcript_source", "duration",
                                         "detected_language", "caption_problem",
                                         "video_id", "title")},
        }

# ------------------------------------------------------------------ results
res = st.session_state.get("result")
if res and res["identity"] == identity:
    if res.get("title"):
        st.subheader(res["title"])
    meta = [f"Transcript: {SOURCE_LABELS.get(res['transcript_source'], '?')}",
            f"length ~{fmt_time(res.get('duration') or 0)}"]
    if res.get("detected_language"):
        meta.append(f"spoken language: {res['detected_language']}")
    st.caption(" · ".join(meta))
    if res.get("caption_problem"):
        st.info(f"Captions weren't available ({res['caption_problem']}), so the audio "
                "was transcribed instead.")

    flagged = {k: r for k, r in (res.get("reviews") or {}).items() if not r["passed"]}
    if flagged:
        st.warning("The critic agent still has concerns about some outputs. "
                   "Double-check them against the video.")
        with st.expander("Critic's remaining concerns"):
            for k, r in flagged.items():
                st.markdown(f"**{config.OUTPUT_TITLES[k]}**")
                for issue in r["issues"]:
                    st.markdown(f"- {issue}")

    outs = res["final_outputs"]
    tab_names = [config.OUTPUT_TITLES[k] for k in outs] + ["Transcript", "Agent trace"]
    tabs = st.tabs(tab_names)
    for tab, (k, text) in zip(tabs, outs.items()):
        with tab:
            st.markdown(text)
    with tabs[-2]:
        st.text_area("Transcript", res.get("transcript_text") or "", height=400,
                     label_visibility="collapsed")
    with tabs[-1]:
        for line in res["trace"]:
            st.markdown(f"- {line}")

    name = f"notes_{res.get('video_id') or 'audio'}"
    cols = st.columns(4)
    cols[0].download_button("⬇️ .md", res["markdown"], f"{name}.md", "text/markdown")
    cols[1].download_button("⬇️ .txt", res["markdown"], f"{name}.txt", "text/plain")
    if res.get("pdf"):
        cols[2].download_button("⬇️ .pdf", res["pdf"], f"{name}.pdf", "application/pdf")
    else:
        cols[2].caption("PDF isn't available for this language; use .md.")
    cols[3].download_button("⬇️ Transcript", res.get("transcript_text") or "",
                            f"{name}_transcript.txt", "text/plain")
