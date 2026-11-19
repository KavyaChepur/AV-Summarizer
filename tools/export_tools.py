"""Export tools: build the notes document as Markdown and PDF."""
import io
import re
from pathlib import Path
from typing import Dict, Optional

from config import OUTPUT_TITLES

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


def build_markdown(title: str, source_line: str, outputs: Dict[str, str],
                   transcript_note: Optional[str] = None) -> str:
    parts = [f"# {title}", "", source_line]
    if transcript_note:
        parts += ["", f"_{transcript_note}_"]
    for key, text in outputs.items():
        parts += ["", f"## {OUTPUT_TITLES.get(key, key)}", "", text.strip()]
    return "\n".join(parts) + "\n"


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


def markdown_to_pdf(markdown: str) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    font, bold = _register_fonts()
    base = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=base["BodyText"], fontName=font,
                          fontSize=10.5, leading=14.5)
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=14, bulletIndent=4)
    heads = {
        1: ParagraphStyle("h1", parent=base["Heading1"], fontName=bold, fontSize=18),
        2: ParagraphStyle("h2", parent=base["Heading2"], fontName=bold, fontSize=14,
                          spaceBefore=10),
        3: ParagraphStyle("h3", parent=base["Heading3"], fontName=bold, fontSize=12),
    }

    story = []
    for raw in markdown.splitlines():
        line = raw.rstrip()
        if not line.strip():
            story.append(Spacer(1, 4))
            continue
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            level = min(len(m.group(1)), 3)
            story.append(Paragraph(_inline(m.group(2), bold), heads[level]))
            continue
        m = re.match(r"^(\s*)[-*+]\s+(.*)", line)
        if m:
            indent = len(m.group(1).replace("\t", "    ")) // 2
            style = ParagraphStyle(f"b{indent}", parent=bullet,
                                   leftIndent=14 + 12 * indent, bulletIndent=4 + 12 * indent)
            story.append(Paragraph(_inline(m.group(2), bold), style, bulletText="•"))
            continue
        m = re.match(r"^\s*(\d+)[.)]\s+(.*)", line)
        if m:
            story.append(Paragraph(_inline(m.group(2), bold), bullet,
                                   bulletText=f"{m.group(1)}."))
            continue
        story.append(Paragraph(_inline(line.strip("_"), bold), body))

    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                      topMargin=16 * mm, bottomMargin=16 * mm,
                      title="AV Summarizer notes").build(story)
    return buf.getvalue()
