"""Local transcript search for the chat agent (BM25 keyword ranking, no API calls).

Used only when a transcript is too long to give to the model in full.
"""
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple

from tools.text_tools import parse_time

_LINE_RE = re.compile(r"^\[(\d{1,2}(?::\d{2}){1,2})\]\s*(.*)$")
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_STOPWORDS = set("""a an and are as at be but by can did do does for from had has have
how i if in into is it its me my of on or our so than that the their them then there
these they this to was we were what when where which who why will with you your about
video talk speaker said say says tell""".split())


@dataclass
class Passage:
    start: float
    text: str        # "[MM:SS] …" lines


def tokenize(text: str) -> List[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if len(w) > 1 and w not in _STOPWORDS]


def make_passages(transcript_text: str, lines_per_passage: int = 4,
                  overlap: int = 1) -> List[Passage]:
    """Group the ~30 s transcript lines into overlapping ~2-minute passages."""
    lines = []
    for raw in transcript_text.splitlines():
        m = _LINE_RE.match(raw.strip())
        if m:
            lines.append((parse_time(m.group(1)), raw.strip()))
    step = max(1, lines_per_passage - overlap)
    passages = []
    for i in range(0, len(lines), step):
        window = lines[i:i + lines_per_passage]
        passages.append(Passage(window[0][0], "\n".join(t for _, t in window)))
        if i + lines_per_passage >= len(lines):
            break
    return passages


class BM25Index:
    def __init__(self, passages: List[Passage], k1: float = 1.5, b: float = 0.75):
        self.passages = passages
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(p.text)) for p in passages]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df = Counter(w for d in self.docs for w in d)
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def search(self, query: str, k: int = 8) -> List[Tuple[float, Passage]]:
        terms = set(tokenize(query))
        scored = []
        for doc, length, passage in zip(self.docs, self.lengths, self.passages):
            score = 0.0
            for t in terms:
                tf = doc.get(t)
                if tf:
                    norm = tf * (self.k1 + 1) / (
                        tf + self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1)))
                    score += self.idf[t] * norm
            if score > 0:
                scored.append((score, passage))
        scored.sort(key=lambda x: -x[0])
        return scored[:k]
