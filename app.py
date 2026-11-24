"""Streamlit UI for the agentic AV summarizer."""
import shutil
import tempfile
from pathlib import Path

import streamlit as st

import config  # loads .env
from agents import chat_agent
from errors import InvalidYouTubeURL, PipelineError
from graph.builder import NODE_LABELS
from graph.runner import stream_pipeline
from tools import mindmap_tools
from tools.audio_tools import AUDIO_EXTENSIONS
from tools.export_tools import PDF_UNSUPPORTED_LANGUAGES, notes_to_pdf
from tools.stt_tools import WHISPER_SIZES
from tools.text_tools import fmt_time, link_timestamps
from tools.youtube_tools import extract_video_id

st.set_page_config(page_title="AV Summarizer AI", page_icon="🎬", layout="centered")
st.title("🎬 AV Summarizer AI")
st.caption("A team of AI agents turns YouTube videos and audio/video files into "
           "summaries, a mind map and key takeaways, and answers your questions about them.")

TITLE_TO_KEY = {v: k for k, v in config.OUTPUT_TITLES.items()}
AUDIO_PREVIEW = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".opus"}
# Rough share of the run each node represents, for the progress bar.
WRITER_PROGRESS = 0.72
NODE_PROGRESS = {"ingest": 0.03, "fetch_captions": 0.08, "download_audio": 0.15,
                 "speech_to_text": 0.45, "preprocess": 0.5, "chunk_notes": 0.62,
                 **{k: WRITER_PROGRESS for k in config.OUTPUT_TITLES},
                 "critic": 0.82, "revise": 0.85, "translate": 0.93, "export": 1.0}
SOURCE_LABELS = {"captions": "YouTube captions", "whisper": "Whisper speech-to-text",
                 "gemini": "Gemini speech-to-text"}
SUGGESTED_QUESTIONS = ["What is the main idea?",
                       "Explain the key points in simple words",
                       "What examples or evidence are given?"]


def show_image(url: str) -> None:
    try:
        st.image(url, width="stretch")
    except TypeError:  # older Streamlit
        st.image(url, use_container_width=True)


def show_page(html_page: str, height: int) -> None:
    """Embed a full HTML page (with scripts) in an iframe."""
    if hasattr(st, "iframe"):          # Streamlit ≥ 1.56
        st.iframe(html_page, height=height)
    else:                              # older Streamlit
        import streamlit.components.v1 as components
        components.html(html_page, height=height, scrolling=False)


def mind_map_tree(res: dict):
    text = res["final_outputs"].get("mind_map")
    if not text:
        return None
    try:
        return mindmap_tools.to_tree(text, fallback_root=res.get("title") or "Main topic")
    except ValueError:
        return None


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Outputs")
    wanted = st.multiselect(
        "Include", list(config.OUTPUT_TITLES.values()),
        default=[config.OUTPUT_TITLES[k] for k in config.DEFAULT_OUTPUTS])
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
            "5. **Short summary, Long summary, Mind map and Takeaways agents** write "
            "in parallel.\n"
            "6. **Critic agent** checks facts and structure and sends drafts back "
            f"for revision (up to {config.MAX_REVISIONS} times).\n"
            "7. **Translator agent** translates if needed, then the notes are exported.\n"
            "8. **Chat agent** answers your questions about the recording.")

# ------------------------------------------------------------------ input
source = st.radio("Source", ["YouTube link", "Upload audio/video"], horizontal=True,
                  label_visibility="collapsed")

identity = None   # identifies the current input, so old results aren't shown for new input
uploaded = None
link = ""

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
    state = {"options": {"outputs": outputs, "language": language,
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
                pdf = notes_to_pdf(final["doc_title"], final["source_line"],
                                   final["final_outputs"], final.get("transcript_note"))
            except Exception:
                pdf = None
        title = final.get("title") or (f"YouTube video {final['video_id']}"
                                       if final.get("video_id") else "Recording")
        st.session_state["result"] = {
            "identity": identity, "trace": trace, "pdf": pdf, "language": language,
            "chat_source": chat_agent.ChatSource(
                title=title, transcript_text=final.get("transcript_text") or "",
                notes=final["final_outputs"], duration=final.get("duration") or 0.0),
            **{k: final.get(k) for k in ("final_outputs", "markdown", "transcript_text",
                                         "reviews", "transcript_source", "duration",
                                         "detected_language", "caption_problem",
                                         "video_id", "title")},
        }

# ------------------------------------------------------------------ results
res = st.session_state.get("result")
if res and res["identity"] == identity:
    vid = res.get("video_id")
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
                st.markdown(f"**{config.OUTPUT_TITLES.get(k, k)}**")
                for issue in r["issues"]:
                    st.markdown(f"- {issue}")

    outs = res["final_outputs"]
    tree = mind_map_tree(res)
    tabs = st.tabs([config.OUTPUT_TITLES.get(k, k) for k in outs] + ["Transcript"])
    for tab, (k, text) in zip(tabs, outs.items()):
        with tab:
            if k == "mind_map" and tree is not None:
                show_page(mindmap_tools.to_html(tree, res.get("title") or "Mind map",
                                                height="540px"), height=560)
                st.caption("Drag to move, scroll to zoom, click a node to fold or unfold it.")
                with st.expander("Mind map as text"):
                    st.markdown(mindmap_tools.to_outline(tree))
            else:
                st.markdown(link_timestamps(text, vid))
    with tabs[-1]:
        st.text_area("Transcript", res.get("transcript_text") or "", height=400,
                     label_visibility="collapsed")

    with st.expander("Agent trace"):
        for line in res["trace"]:
            st.markdown(f"- {line}")

    name = f"notes_{vid or 'audio'}"
    cols = st.columns(5)
    cols[0].download_button("⬇️ .md", res["markdown"], f"{name}.md", "text/markdown")
    cols[1].download_button("⬇️ .txt", res["markdown"], f"{name}.txt", "text/plain")
    if res.get("pdf"):
        cols[2].download_button("⬇️ .pdf", res["pdf"], f"{name}.pdf", "application/pdf")
    else:
        cols[2].caption("No PDF for this language; use .md.")
    if tree is not None:
        cols[3].download_button("⬇️ Mind map", mindmap_tools.to_html(
            tree, res.get("title") or "Mind map"), f"{name}_mindmap.html", "text/html")
    cols[4].download_button("⬇️ Transcript", res.get("transcript_text") or "",
                            f"{name}_transcript.txt", "text/plain")

    # -------------------------------------------------------------- chat
    st.divider()
    st.subheader("💬 Ask about this recording")
    history = st.session_state.setdefault("chats", {}).setdefault(str(identity), [])
    for m in history:
        with st.chat_message(m["role"]):
            st.markdown(link_timestamps(m["content"], vid))

    pending = None
    if not history:
        st.caption("Ask anything about the video or audio, for example:")
        for col, q in zip(st.columns(len(SUGGESTED_QUESTIONS)), SUGGESTED_QUESTIONS):
            if col.button(q, key=f"suggest_{q}", width="stretch"):
                pending = q

    question = st.chat_input("Ask a question about the recording…") or pending
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("The chat agent is reading the transcript…"):
                    reply = chat_agent.answer(question, history, res["chat_source"])
            except PipelineError as e:
                st.error(str(e))
            else:
                st.markdown(link_timestamps(reply.answer, vid))
                if reply.mode == "search":
                    st.caption(f"Long recording: answered from the {len(reply.passages)} "
                               "most relevant transcript passages.")
                history += [{"role": "user", "content": question},
                            {"role": "assistant", "content": reply.answer}]
    if history:
        st.button("Clear chat", on_click=history.clear)
