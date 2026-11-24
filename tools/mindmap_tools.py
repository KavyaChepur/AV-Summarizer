"""Mind map tools: parse the Mind Map agent's outline and render it.

The Mind Map agent writes a nested Markdown bullet list (easy for an LLM to
write, for the critic to check and for the translator to translate). This
module turns that outline into a tree and renders it as:
  * an interactive markmap web page (in the app and as a download)
  * a Mermaid `mindmap` diagram (inside the Markdown notes)
  * a reportlab drawing (inside the PDF)
"""
import html
import re
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

import config

_ITEM_RE = re.compile(r"^(\s*)(?:[-*+]|\d+[.)])\s+(.*\S)\s*$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_TS_RE = re.compile(r"\[?\b\d{1,2}(?::\d{2}){1,2}\b\]?")

MAX_LABEL_CHARS = 80


@dataclass
class MindNode:
    label: str
    children: List["MindNode"] = field(default_factory=list)

    def depth(self) -> int:
        return 1 + max((c.depth() for c in self.children), default=0)

    def walk(self) -> Iterator["MindNode"]:
        yield self
        for c in self.children:
            yield from c.walk()

    def leaves(self) -> int:
        return sum(c.leaves() for c in self.children) if self.children else 1


def clean_label(text: str) -> str:
    text = re.sub(r"\*\*|__|`", "", text)
    text = _TS_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip(" -:–—")
    return text


def parse_outline(text: str) -> List[MindNode]:
    """Nested Markdown list (and/or headings) -> list of top-level nodes."""
    roots: List[MindNode] = []
    stack: List[Tuple[int, MindNode]] = []   # (indent, node)
    in_fence = False
    for raw in (text or "").splitlines():
        line = raw.expandtabs(4).rstrip()
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not line.strip():
            continue
        m = _HEADING_RE.match(line)
        if m:   # headings sit above any list item: "#" < "##" < list items
            indent, label = -100 + len(m.group(1)), m.group(2)
        else:
            m = _ITEM_RE.match(line)
            if not m:
                continue   # stray prose: ignore
            indent, label = len(m.group(1)), m.group(2)
        label = clean_label(label)
        if not label:
            continue
        node = MindNode(label)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        (stack[-1][1].children if stack else roots).append(node)
        stack.append((indent, node))
    return roots


def to_tree(text: str, fallback_root: str = "Main topic") -> MindNode:
    """Outline -> single-rooted tree. Raises ValueError if there's no outline."""
    roots = parse_outline(text)
    if not roots:
        raise ValueError("no bullet outline found")
    return roots[0] if len(roots) == 1 else MindNode(fallback_root, roots)


def validate_outline(text: str) -> List[str]:
    """Code checks used by the critic. Returns a list of issues (empty = fine)."""
    roots = parse_outline(text)
    if not roots:
        return ["The mind map must be a nested Markdown bullet list (central topic first, "
                "then indented branches)."]
    issues = []
    if len(roots) > 1:
        issues.append(f"The mind map has {len(roots)} top-level items. It needs exactly one "
                      "central topic, with every main theme indented under it.")
    root = to_tree(text)
    if len(root.children) < 2:
        issues.append("Give the central topic at least 2 main branches (the main themes).")
    elif len(root.children) > 9:
        issues.append(f"There are {len(root.children)} main branches; group them into at "
                      "most 8 themes.")
    if root.depth() > 5:
        issues.append("The mind map is too deep; use at most 4 levels.")
    long_labels = [n.label for n in root.walk() if len(n.label) > MAX_LABEL_CHARS]
    if long_labels:
        issues.append("Keep every node a short phrase (under ~8 words), not a sentence. "
                      "Too long: " + "; ".join(f"'{l[:50]}…'" for l in long_labels[:3]))
    return issues


def to_outline(root: MindNode) -> str:
    lines = []

    def rec(n: MindNode, level: int):
        lines.append(f"{'  ' * level}- {n.label}")
        for c in n.children:
            rec(c, level + 1)
    rec(root, 0)
    return "\n".join(lines)


# ---------------------------------------------------------------- Mermaid
def _mermaid_text(label: str) -> str:
    # Mermaid mindmap syntax treats brackets/parens/quotes as shape markers.
    return re.sub(r"[()\[\]{}\"`<>]", " ", label).strip() or "…"


def to_mermaid(root: MindNode) -> str:
    lines = ["mindmap", f"  root(({_mermaid_text(root.label)}))"]

    def rec(n: MindNode, level: int):
        for c in n.children:
            lines.append(f"{'  ' * level}{_mermaid_text(c.label)}")
            rec(c, level + 1)
    rec(root, 2)
    return "\n".join(lines)


# ---------------------------------------------------------------- markmap HTML
def to_markmap_markdown(root: MindNode) -> str:
    out = [f"# {root.label}"]

    def rec(n: MindNode, level: int):
        for c in n.children:
            out.append(f"{'  ' * level}- {c.label}")
            rec(c, level + 1)
    rec(root, 0)
    return "\n".join(out)


def to_html(root: MindNode, title: str = "Mind map", height: str = "100vh") -> str:
    """Standalone interactive mind map page (zoom, pan, click to fold branches)."""
    md = to_markmap_markdown(root).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
  html, body {{ margin: 0; padding: 0; background: #ffffff; overflow: hidden; }}
  .markmap {{ position: relative; display: block; width: 100%; height: {height}; }}
  .markmap > svg {{ width: 100%; height: 100%; }}
</style>
<script>window.markmap = {{ autoLoader: {{ toolbar: true }} }};</script>
</head>
<body>
<div class="markmap"><script type="text/template">
---
markmap:
  initialExpandLevel: 3
  maxWidth: 260
---
{md}
</script></div>
<script src="{config.MARKMAP_AUTOLOADER_URL}"></script>
</body>
</html>
"""


# ---------------------------------------------------------------- PDF drawing
_BRANCH_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756",
                  "#72B7B2", "#9D755D", "#EECA3B"]


def to_drawing(root: MindNode, width: float, max_height: float,
               font: str = "Helvetica", bold_font: Optional[str] = None):
    """A reportlab Drawing of the mind map as a left-to-right tree."""
    from reportlab.graphics.shapes import Drawing, PolyLine, Rect, String
    from reportlab.lib import colors
    from reportlab.pdfbase.pdfmetrics import stringWidth

    bold_font = bold_font or font
    levels = root.depth()
    gap = 16.0
    col_w = width / levels
    box_w = col_w - gap
    size, leading, pad = 7.5, 9.0, 3.0

    def wrap(text: str, f: str) -> List[str]:
        words, lines, cur = text.split(), [], ""
        for w in words:
            trial = f"{cur} {w}".strip()
            if stringWidth(trial, f, size) <= box_w - 2 * pad or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = w
        return (lines + [cur]) if cur else lines or [""]

    # 1) measure; 2) stack leaves top-down; parents centre on their children
    placed = []   # (node, depth, y_center, lines, height, color)
    cursor = [0.0]

    def layout(n: MindNode, depth: int, color: str) -> float:
        f = bold_font if depth <= 1 else font
        lines = wrap(n.label, f)
        h = len(lines) * leading + 2 * pad
        if n.children:
            centers = [layout(c, depth + 1, _BRANCH_COLORS[i % len(_BRANCH_COLORS)]
                              if depth == 0 else color)
                       for i, c in enumerate(n.children)]
            y = (centers[0] + centers[-1]) / 2
        else:
            y = cursor[0] + h / 2
            cursor[0] += h + 5
        placed.append((n, depth, y, lines, h, color, f))
        return y

    layout(root, 0, "#333333")
    total_h = cursor[0]

    d = Drawing(width, total_h)
    pos = {id(n): (depth, y, h) for n, depth, y, _, h, _, _ in placed}
    color_of = {id(n): color for n, _, _, _, _, color, _ in placed}

    def flip(y: float) -> float:   # layout grows downwards, PDF y grows upwards
        return total_h - y

    for n, depth, y, lines, h, color, f in placed:   # connectors first
        for c in n.children:
            cd, cy, _ = pos[id(c)]
            x1, x2 = depth * col_w + box_w, cd * col_w
            mid = (x1 + x2) / 2
            d.add(PolyLine([x1, flip(y), mid, flip(y), mid, flip(cy), x2, flip(cy)],
                           strokeColor=colors.HexColor(color_of[id(c)]),
                           strokeWidth=1))
    for n, depth, y, lines, h, color, f in placed:
        x = depth * col_w
        c = colors.HexColor(color)
        if depth <= 1:   # central topic and main branches: filled
            fill, stroke, text = c, c, colors.white
        else:
            fill, stroke, text = colors.white, c, colors.HexColor("#222222")
        d.add(Rect(x, flip(y) - h / 2, box_w, h, rx=4, ry=4, fillColor=fill,
                   strokeColor=stroke, strokeWidth=1))
        top = flip(y) + h / 2 - pad - size
        for i, line in enumerate(lines):
            d.add(String(x + pad, top - i * leading + 1, line, fontName=f,
                         fontSize=size, fillColor=text))

    if total_h > max_height:   # shrink to fit one page
        s = max_height / total_h
        d.scale(s, s)
        d.width, d.height = width * s, max_height
    return d
