"""Command-line version of the app.

Examples:
  python run_cli.py "https://youtu.be/VIDEO_ID"
  python run_cli.py lecture.mp3 --language Hindi -o notes.md --chat
  python run_cli.py "https://youtu.be/VIDEO_ID" --force-stt --stt-engine gemini
  python run_cli.py --graph            # print the agent graph as Mermaid
"""
import argparse
import sys
from pathlib import Path

import config
from agents import chat_agent
from errors import PipelineError
from graph.builder import NODE_LABELS, get_graph
from graph.runner import stream_pipeline
from tools import mindmap_tools


def main() -> int:
    p = argparse.ArgumentParser(description="Agentic YouTube / audio summarizer")
    p.add_argument("source", nargs="?", help="YouTube link or path to an audio/video file")
    p.add_argument("--outputs", default=",".join(config.DEFAULT_OUTPUTS),
                   help="comma-separated, any of: " + ",".join(config.OUTPUT_TITLES))
    p.add_argument("--language", default="English", choices=config.LANGUAGES)
    p.add_argument("--stt-engine", choices=["whisper", "gemini"], default=config.STT_ENGINE)
    p.add_argument("--whisper-model", default=config.WHISPER_MODEL,
                   choices=["tiny", "base", "small", "medium"])
    p.add_argument("--force-stt", action="store_true",
                   help="YouTube: ignore captions and transcribe the audio")
    p.add_argument("--no-fallback", action="store_true",
                   help="YouTube: don't transcribe the audio when captions are missing")
    p.add_argument("--no-critic", action="store_true", help="skip the critic agent")
    p.add_argument("-o", "--out", help="write the Markdown notes to this file "
                                       "(and the mind map to <name>_mindmap.html)")
    p.add_argument("--chat", action="store_true",
                   help="after the notes, ask questions about the recording")
    p.add_argument("--graph", action="store_true", help="print the graph as Mermaid and exit")
    args = p.parse_args()

    if args.graph:
        print(get_graph().get_graph().draw_mermaid())
        return 0
    if not args.source:
        p.error("source is required")

    outputs = [o.strip() for o in args.outputs.split(",") if o.strip()]
    bad = set(outputs) - set(config.OUTPUT_TITLES)
    if bad or not outputs:
        p.error(f"unknown outputs: {', '.join(bad) or '(none given)'}")

    state = {
        "options": {
            "outputs": outputs, "language": args.language,
            "stt_engine": args.stt_engine, "whisper_model": args.whisper_model,
            "force_stt": args.force_stt, "allow_stt_fallback": not args.no_fallback,
            "use_critic": not args.no_critic,
        }
    }
    if Path(args.source).is_file():
        state.update(source_type="upload", file_path=str(Path(args.source).resolve()),
                     file_name=Path(args.source).name)
    else:
        state.update(source_type="youtube", url=args.source)

    final = None
    try:
        for ev in stream_pipeline(state):
            if ev["type"] == "progress":
                print(f"   … {ev['message']}", file=sys.stderr)
            elif ev["type"] == "step":
                for line in ev["log"]:
                    print(f"[{NODE_LABELS.get(ev['node'], ev['node'])}] {line}", file=sys.stderr)
            else:
                final = ev["state"]
    except PipelineError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    for k, r in (final.get("reviews") or {}).items():
        if not r["passed"]:
            print(f"⚠️  The critic still flagged '{k}':", file=sys.stderr)
            for issue in r["issues"]:
                print(f"     - {issue}", file=sys.stderr)

    title = final.get("title") or final.get("video_id") or "Recording"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(final["markdown"], encoding="utf-8")
        print(f"Saved notes to {out}", file=sys.stderr)
        if final["final_outputs"].get("mind_map"):
            try:
                tree = mindmap_tools.to_tree(final["final_outputs"]["mind_map"], title)
                html_path = out.with_name(f"{out.stem}_mindmap.html")
                html_path.write_text(mindmap_tools.to_html(tree, title), encoding="utf-8")
                print(f"Saved the interactive mind map to {html_path}", file=sys.stderr)
            except ValueError:
                pass
    else:
        print(final["markdown"])

    if args.chat:
        chat(chat_agent.ChatSource(title=title,
                                   transcript_text=final.get("transcript_text") or "",
                                   notes=final["final_outputs"],
                                   duration=final.get("duration") or 0.0))
    return 0


def chat(source) -> None:
    print("\nAsk questions about the recording (empty line or 'exit' to quit).",
          file=sys.stderr)
    history = []
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in {"exit", "quit"}:
            break
        try:
            reply = chat_agent.answer(question, history, source)
        except PipelineError as e:
            print(f"Error: {e}", file=sys.stderr)
            continue
        print(f"\nAssistant: {reply.answer}")
        history += [{"role": "user", "content": question},
                    {"role": "assistant", "content": reply.answer}]


if __name__ == "__main__":
    sys.exit(main())
