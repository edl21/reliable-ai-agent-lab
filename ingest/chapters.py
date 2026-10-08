"""Chapter map construction for the public-domain demo corpus.

The map assigns a dense application index to each detected chapter in
reading order. ``max_chapter`` refers to this application index rather
than a printed page number. A narrative prologue or epilogue is included
when present.

Output: ``corpus/chapter_map.json`` — a list where index i-1 has::

    {
        "index": i,                    # 1-based
        "source_filename": <str>,
        "original_label": <str>,       # "Chapter 1" or "Epilogue"
        "chapter_title": <str>,        # "Loomings"
        "pdf_page_start": <int>,       # 1-based
        "pdf_page_end": <int>,         # 1-based, inclusive
    }
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from ingest.extract import ExtractedPage, load_pages
from ingest.fetch import CORPUS_DIR, CORPUS_FILENAME


CHAPTER_MAP_JSON: Path = CORPUS_DIR / "chapter_map.json"

# Chapter headings can use arabic or roman numerals and may include the
# title on the same line.
_CHAPTER_HEADING = re.compile(
    r"^\s*chapter\s+(?P<num>[IVXLCDM]+|\d+)\b\s*[.:]?\s*(?P<title>.*)?$",
    re.IGNORECASE,
)
_PROLOGUE_HEADING = re.compile(r"^\s*prologue\b", re.IGNORECASE)
_EPILOGUE_HEADING = re.compile(r"^\s*epilogue\b", re.IGNORECASE)

_ROMAN: dict[str, int] = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


@dataclass
class ChapterEntry:
    """One row in the chapter map."""

    index: int
    source_filename: str
    original_label: str
    chapter_title: str
    pdf_page_start: int
    pdf_page_end: int

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "source_filename": self.source_filename,
            "original_label": self.original_label,
            "chapter_title": self.chapter_title,
            "pdf_page_start": self.pdf_page_start,
            "pdf_page_end": self.pdf_page_end,
        }


class ChapterMapError(RuntimeError):
    """Raised when the chapter detector produces an obviously-wrong map
    (e.g. fewer than 15 chapters for the configured long-form corpus)."""


# --- public API -----------------------------------------------------------


def build_chapter_map(*, pages: list[ExtractedPage] | None = None) -> list[ChapterEntry]:
    """Detect chapter boundaries in the extracted pages, assign 1..N
    in reading order (Prologue first), and persist ``chapter_map.json``."""
    pages = pages if pages is not None else load_pages()
    boundaries = _detect_boundaries(pages)

    if len(boundaries) < 15:
        raise ChapterMapError(
            f"detected {len(boundaries)} sections; expected at least 15. "
            "Chapter-heading pattern is not matching this PDF's layout."
        )

    entries: list[ChapterEntry] = []
    for i, (start_page, original_label, title) in enumerate(boundaries):
        end_page = (
            boundaries[i + 1][0] - 1
            if i + 1 < len(boundaries)
            else pages[-1].pdf_page
        )
        entries.append(
            ChapterEntry(
                index=i + 1,
                source_filename=CORPUS_FILENAME,
                original_label=original_label,
                chapter_title=title,
                pdf_page_start=start_page,
                pdf_page_end=end_page,
            )
        )

    _write_map(entries)
    return entries


def load_chapter_map() -> list[ChapterEntry]:
    """Read back the chapter map."""
    if not CHAPTER_MAP_JSON.exists():
        raise ChapterMapError(
            f"{CHAPTER_MAP_JSON} missing; run `python -m ingest.build` first"
        )
    data = json.loads(CHAPTER_MAP_JSON.read_text(encoding="utf-8"))
    return [ChapterEntry(**row) for row in data]


def pdf_page_to_chapter_index(
    pdf_page: int, chapter_map: list[ChapterEntry]
) -> int | None:
    """Map a PDF page number to its ``index`` in the chapter map, or
    None if the page falls in a gap (should not happen for narrative
    pages after extraction)."""
    for entry in chapter_map:
        if entry.pdf_page_start <= pdf_page <= entry.pdf_page_end:
            return entry.index
    return None


# --- internals -----------------------------------------------------------


def _detect_boundaries(
    pages: list[ExtractedPage],
) -> list[tuple[int, str, str]]:
    """Return ``(pdf_page, original_label, chapter_title)`` boundaries.

    Headings are detected from raw extracted lines rather than the
    cleaned page text. This preserves the title/body boundary even
    when cleanup joins hard line breaks inside paragraphs.
    """
    out: list[tuple[int, str, str]] = []
    seen_labels: set[str] = set()

    for page in pages:
        lines = [line.strip() for line in page.raw.splitlines() if line.strip()]
        for line_index, line in enumerate(lines):
            if _PROLOGUE_HEADING.fullmatch(line) and "Prologue" not in seen_labels:
                out.append((page.pdf_page, "Prologue", "Prologue"))
                seen_labels.add("Prologue")
                break
            if _EPILOGUE_HEADING.fullmatch(line) and "Epilogue" not in seen_labels:
                out.append((page.pdf_page, "Epilogue", "Epilogue"))
                seen_labels.add("Epilogue")
                break

            match = _CHAPTER_HEADING.fullmatch(line)
            if not match:
                continue
            number = _to_int(match.group("num"))
            if number <= 0:
                continue
            label = f"Chapter {number}"
            if label in seen_labels:
                continue
            raw_title = (match.group("title") or "").strip(" .:—-")
            if not raw_title and line_index + 1 < len(lines):
                raw_title = lines[line_index + 1]
            title = _trim_title(raw_title)
            out.append((page.pdf_page, label, title))
            seen_labels.add(label)
            break

    return out


def _trim_title(raw: str) -> str:
    """Extract a concise title from either title case or all-caps text.

    The all-caps path also supports older PDFs where cleanup joins the
    heading to the opening sentence.
    """
    if not raw:
        return ""
    raw = raw.strip()
    first = raw.split()[0].strip(".,:;!?'’“”—–-\u2018\u2019\u201c\u201d")
    first_alpha = [ch for ch in first if ch.isalpha()]
    if first_alpha and not all(ch.isupper() for ch in first_alpha):
        return raw.rstrip(" .:—-")
    words = raw.split()
    kept: list[str] = []
    for word in words:
        # Strip surrounding punctuation for the case check, then
        # require every alphabetic character to be uppercase.
        core = word.strip(".,:;!?'’“”—–-\u2018\u2019\u201c\u201d")
        alpha = [ch for ch in core if ch.isalpha()]
        if not alpha:
            # No letters — punctuation-only token; keep and continue.
            kept.append(word)
            continue
        if all(ch.isupper() for ch in alpha):
            kept.append(word)
            continue
        break
    title = " ".join(kept).strip(" .:—-\u2018\u2019\u201c\u201d")
    return title or raw.rstrip(" .:—-")


def _to_int(token: str) -> int:
    """Accept either arabic ('12') or roman ('XII') numerals."""
    token = token.strip().upper()
    if token.isdigit():
        return int(token)
    total = 0
    prev = 0
    for ch in reversed(token):
        val = _ROMAN.get(ch, 0)
        if val < prev:
            total -= val
        else:
            total += val
            prev = val
    return total


def _roman(n: int | None) -> str:
    """Render an int back as a roman numeral for pretty labelling."""
    if n is None or n <= 0:
        return "?"
    pairs = [
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
        (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
        (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
    ]
    out = ""
    for value, symbol in pairs:
        while n >= value:
            out += symbol
            n -= value
    return out


def _write_map(entries: list[ChapterEntry]) -> None:
    CHAPTER_MAP_JSON.parent.mkdir(parents=True, exist_ok=True)
    CHAPTER_MAP_JSON.write_text(
        json.dumps([e.to_dict() for e in entries], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "CHAPTER_MAP_JSON",
    "ChapterEntry",
    "ChapterMapError",
    "build_chapter_map",
    "load_chapter_map",
    "pdf_page_to_chapter_index",
]
