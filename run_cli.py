"""Command-line version of the app.

Examples:
  python run_cli.py "https://youtu.be/VIDEO_ID"
  python run_cli.py lecture.mp3 --length Detailed --language Hindi
  python run_cli.py "https://youtu.be/VIDEO_ID" --force-stt --stt-engine gemini
  python run_cli.py --graph            # print the agent graph as Mermaid
"""
import argparse
import sys
from pathlib import Path

import config
from errors import PipelineError
from graph.builder import NODE_LABELS, get_graph
from graph.runner import stream_pipeline


def main() -> int:
    p = argparse.ArgumentParser(description="Agentic YouTube / audio summarizer")
    p.add_argument("source", nargs="?", help="YouTube link or path to an audio/video file")
    p.add_argument("--length", choices=list(config.LENGTHS), default="Medium")
    p.add_argument("--outputs", default="summary,takeaways,timeline",
                   help="comma-separated: summary,takeaways,timeline")
    p.add_argument("--language", default="English", choices=config.LANGUAGES)
    p.add_argument("--stt-engine", choices=["whisper", "gemini"], default=config.STT_ENGINE)
    p.add_argument("--whisper-model", default=config.WHISPER_MODEL,
                   choices=["tiny", "base", "small", "medium"])
    p.add_argument("--force-stt", action="store_true",
                   help="YouTube: ignore captions and transcribe the audio")
    p.add_argument("--no-fallback", action="store_true",
                   help="YouTube: don't transcribe the audio when captions are missing")
    p.add_argument("--no-critic", action="store_true", help="skip the critic agent")
    p.add_argument("-o", "--out", help="write the Markdown notes to this file")
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
            "length": args.length, "outputs": outputs, "language": args.language,
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

    if args.out:
        Path(args.out).write_text(final["markdown"], encoding="utf-8")
        print(f"Saved notes to {args.out}", file=sys.stderr)
    else:
        print(final["markdown"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
