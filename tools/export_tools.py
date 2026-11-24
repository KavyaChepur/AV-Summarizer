"""Export tools: build the notes document as Markdown and PDF."""
import io
import re
from pathlib import Path
from typing import Dict, List, Optional

from config import OUTPUT_TITLES
from tools import mindmap_tools

# Languages whose scripts reportlab cannot shape correctly in a PDF.
PDF_UNSUPPORTED_LANGUAGES = {"Hindi", "Telugu"}

_FONT_CANDIDATES = [
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/usr/share/fonts/truetype/freefont/FreeSans.ttf",
     "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"),
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ("/Library/Fonts/Arial Unicode.ttf", "/Library/Fonts/Arial Unicode.ttf"),
]


def _mind_map_markdown(outline: str, title: str) -> str:
    """The outline plus a Mermaid diagram (renders on GitHub, Obsidian, VS Code…)."""
    try:
        tree = mindmap_tools.to_tree(outline, fallback_root=title)
    except ValueError:
        return outline.strip()
    return (mindmap_tools.to_outline(tree) + "\n\n```mermaid\n"
            + mindmap_tools.to_mermaid(tree) + "\n```")


def build_markdown(title: str, source_line: str, outputs: Dict[str, str],
                   transcript_note: Optional[str] = None) -> str:
    parts = [f"# {title}", "", source_line]
    if transcript_note:
        parts += ["", f"_{transcript_note}_"]
    for key, text in outputs.items():
        body = _mind_map_markdown(text, title) if key == "mind_map" else text.strip()
        parts += ["", f"## {OUTPUT_TITLES.get(key, key)}", "", body]
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------- PDF
def _register_fonts() -> tuple:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    for regular, bold in _FONT_CANDIDATES:
        if Path(regular).exists():
            try:
                pdfmetrics.registerFont(TTFont("NotesFont", regular))
                pdfmetrics.registerFont(
                    TTFont("NotesFont-Bold", bold if Path(bold).exists() else regular))
                return "NotesFont", "NotesFont-Bold"
            except Exception:
                continue
    return "Helvetica", "Helvetica-Bold"


def _inline(text: str, bold_font: str) -> str:
    """Escape for reportlab's mini-HTML and convert **bold** / *italic*."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", rf'<font name="{bold_font}">\1</font>', text)
    text = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<i>\1</i>", text)
    return text


def _styles(font: str, bold: str) -> dict:
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet

    base = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=base["BodyText"], fontName=font,
                          fontSize=10.5, leading=14.5)
    return {
        "body": body,
        "bullet": ParagraphStyle("bullet", parent=body, leftIndent=14, bulletIndent=4),
        1: ParagraphStyle("h1", parent=base["Heading1"], fontName=bold, fontSize=18),
        2: ParagraphStyle("h2", parent=base["Heading2"], fontName=bold, fontSize=14,
                          spaceBefore=10),
        3: ParagraphStyle("h3", parent=base["Heading3"], fontName=bold, fontSize=12),
    }


def _markdown_story(markdown: str, styles: dict, bold: str) -> List:
    """Markdown -> reportlab flowables (headings, bullets, numbers, paragraphs)."""
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph, Spacer

    story = []
    in_fence = False
    for raw in markdown.splitlines():
        line = raw.rstrip()
        if line.strip().startswith("```"):
            in_fence = not in_fence   # code/mermaid blocks are left out of the PDF
            continue
        if in_fence:
            continue
        if not line.strip():
            story.append(Spacer(1, 4))
            continue
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            level = min(len(m.group(1)), 3)
            story.append(Paragraph(_inline(m.group(2), bold), styles[level]))
            continue
        m = re.match(r"^(\s*)[-*+]\s+(.*)", line)
        if m:
            indent = len(m.group(1).replace("\t", "    ")) // 2
            style = ParagraphStyle(f"b{indent}", parent=styles["bullet"],
                                   leftIndent=14 + 12 * indent, bulletIndent=4 + 12 * indent)
            story.append(Paragraph(_inline(m.group(2), bold), style, bulletText="•"))
            continue
        m = re.match(r"^\s*(\d+)[.)]\s+(.*)", line)
        if m:
            story.append(Paragraph(_inline(m.group(2), bold), styles["bullet"],
                                   bulletText=f"{m.group(1)}."))
            continue
        story.append(Paragraph(_inline(line.strip("_"), bold), styles["body"]))
    return story


def _build(story: List, title: str) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate

    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                      topMargin=16 * mm, bottomMargin=16 * mm, title=title).build(story)
    return buf.getvalue()


def markdown_to_pdf(markdown: str) -> bytes:
    font, bold = _register_fonts()
    return _build(_markdown_story(markdown, _styles(font, bold), bold), "AV Summarizer notes")


def notes_to_pdf(title: str, source_line: str, outputs: Dict[str, str],
                 transcript_note: Optional[str] = None) -> bytes:
    """The notes as a PDF, with the mind map drawn as a diagram."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, Spacer

    font, bold = _register_fonts()
    styles = _styles(font, bold)
    header = f"# {title}\n\n{source_line}\n" + (f"\n_{transcript_note}_\n" if transcript_note else "")
    story = _markdown_story(header, styles, bold)
    width = A4[0] - 36 * mm
    max_height = A4[1] - 32 * mm - 40   # leave room for the section heading
    for key, text in outputs.items():
        story.append(Paragraph(_inline(OUTPUT_TITLES.get(key, key), bold), styles[2]))
        if key == "mind_map":
            try:
                tree = mindmap_tools.to_tree(text, fallback_root=title)
                story += [Spacer(1, 4), mindmap_tools.to_drawing(
                    tree, width, max_height, font=font, bold_font=bold)]
                continue
            except ValueError:
                pass   # not an outline: fall back to the text
        story += _markdown_story(text.strip(), styles, bold)
    return _build(story, title)
