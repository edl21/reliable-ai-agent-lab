"""PDF text extraction and narrative cleanup.

Pipeline (documented per source requirement):

1. **Per-page extraction.** ``pypdf`` extracts text from every page.
   We keep the 1-based PDF page number alongside every surviving line
   because ``/answer``'s citations must carry ``pdf_pages`` per source.

2. **Frequency-based header/footer stripping.** Any short line that
   appears on many pages is treated as boilerplate (running head,
   page number, publisher slug) and dropped. Threshold: appears on
   >= 15% of pages OR is a bare page number.

3. **Front/back matter removal.** Pages before the first chapter are
   excluded. Table of Contents, indexes, copyright pages, and other
   non-narrative material are dropped by title-heading detection.

4. **Line-artefact handling.**
   - ``foo-\\nbar`` → ``foobar`` (hyphen-broken words).
   - Hard line breaks *inside* paragraphs collapse to a single space.
   - Paragraph breaks (blank line separators) are preserved.

The output is a JSONL file, ``corpus/pages.jsonl``, one line per
surviving page containing::

    {"pdf_page": <1-based>, "raw": "<raw text>", "cleaned": "<cleaned text>"}

Downstream (``chapters.py``, ``chunk.py``) reads this file — never the
PDF again.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from ingest.fetch import CORPUS_DIR, CORPUS_FILENAME, load_manifest, update_manifest


PAGES_JSONL: Path = CORPUS_DIR / "pages.jsonl"

# Fraction of pages a short line must appear on to be treated as a
# recurring header/footer. Tuned by inspection: proper narrative lines
# don't recur across chapters, but "THE FELLOWSHIP OF THE RING" (running
# head), publisher slugs, and bare page numbers repeat on many pages.
_HEADER_FOOTER_THRESHOLD: float = 0.15

# A line shorter than this is a candidate for header/footer treatment
# regardless of frequency. Narrative sentences are typically longer.
_SHORT_LINE_CHARS: int = 40

# Headings whose page (and following few pages) we strip out entirely.
# Case-insensitive, whole-page match (after stripping whitespace).
_FRONT_MATTER_HEADINGS: frozenset[str] = frozenset(
    {
        "contents",
        "table of contents",
        "acknowledgements",
        "note on the text",
        "note on the 50th anniversary edition",
        "note on the text of the fiftieth-anniversary edition",
        "foreword",
        "foreword to the second edition",
    }
)
_BACK_MATTER_HEADINGS: frozenset[str] = frozenset(
    {
        "index",
        "index of persons places and things",
        "appendix",
        "maps",
        "about the publisher",
        "copyright",
    }
)


@dataclass
class ExtractedPage:
    """One page of surviving narrative text."""

    pdf_page: int
    raw: str
    cleaned: str


class ExtractionError(RuntimeError):
    """Raised when the PDF cannot be read or produces no text."""


# --- public API -----------------------------------------------------------


def extract_pdf(*, pdf_path: Path | None = None) -> list[ExtractedPage]:
    """Run the full extraction pipeline. Returns cleaned pages and
    writes them to ``corpus/pages.jsonl``. Also updates the manifest
    with ``pdf_pages`` and ``cleaned_word_count``."""
    path = pdf_path or (CORPUS_DIR / CORPUS_FILENAME)
    if not path.exists():
        raise ExtractionError(
            f"{path} missing; run `python -m ingest.build` first"
        )

    reader = PdfReader(str(path))
    pdf_page_count = len(reader.pages)
    if pdf_page_count == 0:
        raise ExtractionError("PDF has zero pages")

    raw_pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        raw_pages.append((i, text))

    # Step 1: identify recurring short lines to drop (headers/footers).
    recurring = _identify_recurring_short_lines(raw_pages, pdf_page_count)

    # Step 2: figure out the start/end page window (drop front/back matter).
    start_page, end_page = _find_narrative_window(raw_pages)

    # Steps 3–4: clean each surviving page.
    kept: list[ExtractedPage] = []
    for pdf_page, raw in raw_pages:
        if pdf_page < start_page or pdf_page > end_page:
            continue
        cleaned = _clean_page_text(raw, drop_lines=recurring)
        if not cleaned.strip():
            continue
        kept.append(ExtractedPage(pdf_page=pdf_page, raw=raw, cleaned=cleaned))

    if not kept:
        raise ExtractionError("no pages survived cleanup — pipeline is broken")

    # Persist.
    _write_pages(kept)

    # Update manifest.
    total_words = sum(len(p.cleaned.split()) for p in kept)
    update_manifest(pdf_pages=pdf_page_count, cleaned_word_count=total_words)
    return kept


def load_pages() -> list[ExtractedPage]:
    """Read back the JSONL produced by ``extract_pdf``."""
    if not PAGES_JSONL.exists():
        raise ExtractionError(
            f"{PAGES_JSONL} missing; run `python -m ingest.build` first"
        )
    out: list[ExtractedPage] = []
    with PAGES_JSONL.open("r", encoding="utf-8") as fh:
        for line in fh:
            data = json.loads(line)
            out.append(ExtractedPage(**data))
    return out


# --- internals -----------------------------------------------------------


def _identify_recurring_short_lines(
    raw_pages: list[tuple[int, str]], total_pages: int
) -> frozenset[str]:
    """Any short line appearing on more than ``_HEADER_FOOTER_THRESHOLD``
    of pages is treated as a running header/footer/publisher slug.
    Bare page numbers are always stripped regardless of length."""
    counter: Counter[str] = Counter()
    for _, text in raw_pages:
        seen_on_this_page: set[str] = set()
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if len(stripped) > _SHORT_LINE_CHARS:
                continue
            # Count once per page — a header that appears twice on one
            # page should not double-weight it.
            if stripped in seen_on_this_page:
                continue
            seen_on_this_page.add(stripped)
            counter[stripped] += 1

    limit = max(3, int(_HEADER_FOOTER_THRESHOLD * total_pages))
    recurring = {line for line, count in counter.items() if count >= limit}
    return frozenset(recurring)


def _find_narrative_window(raw_pages: list[tuple[int, str]]) -> tuple[int, int]:
    """Return the (start_page, end_page) of narrative content, dropping
    front matter (contents, acknowledgements, notes) and back matter
    (maps, works lists, copyright and publisher pages)."""
    start = 1
    end = raw_pages[-1][0] if raw_pages else 1

    # Walk forward until we see a prologue or the first chapter as a
    # standalone heading on its own line. Searching the whole page
    # would match the table of contents and retain the front matter.
    for pdf_page, text in raw_pages:
        opening = "\n".join(text.splitlines()[:8])
        if re.search(r"(?im)^\s*prologue\b", opening) or re.search(
            r"(?im)^\s*chapter\s+(?:i|1)\b", opening
        ):
            start = pdf_page
            break

    # Find the first page after the narrative whose opening lines are
    # unmistakable back matter. Looking forward is important here:
    # the copyright/publisher pages are separated by blank/map pages,
    # so reverse-scanning stops too early on a page containing only
    # punctuation.
    for pdf_page, text in raw_pages:
        if pdf_page <= start:
            continue
        opening = "\n".join(text.splitlines()[:4]).strip().lower()
        if re.search(
            r"(?im)^\s*(?:maps|about the publisher|copyright)\b",
            opening,
        ) or re.search(r"(?im)^\s*(?:works by|ii\s*\n\s*works by)\b", opening):
            end = pdf_page - 1
            break

    if start > end:
        # Fallback: use the whole document. Front/back matter is
        # imperfect but we would rather over-include than exclude.
        start, end = 1, raw_pages[-1][0]
    return start, end


def _clean_page_text(text: str, *, drop_lines: frozenset[str]) -> str:
    """Apply the line-artefact cleanup to one page's raw text."""
    # 1. Drop recurring lines (headers/footers) and bare page numbers.
    out_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            out_lines.append("")  # preserve paragraph breaks
            continue
        if stripped in drop_lines:
            continue
        if _looks_like_page_number(stripped):
            continue
        # Skip standalone front-matter headings that snuck through.
        if stripped.lower() in _FRONT_MATTER_HEADINGS:
            continue
        out_lines.append(stripped)

    # 2. Rejoin hyphen-broken words. We do this on the whole text so
    #    the hyphen-and-newline sits between two adjacent chars.
    joined = "\n".join(out_lines)
    joined = re.sub(r"(\w)-\n(\w)", r"\1\2", joined)

    # 3. Collapse hard line breaks inside paragraphs to a space, while
    #    keeping paragraph breaks (double newlines).
    #    "\n\n" or more → PARA marker; single "\n" → " "; then unmark.
    para_marker = "\x00\x00"
    tmp = re.sub(r"\n{2,}", para_marker, joined)
    tmp = re.sub(r"\n+", " ", tmp)
    tmp = tmp.replace(para_marker, "\n\n")

    # 4. Collapse multiple spaces produced by the collapse step.
    tmp = re.sub(r"[ \t]{2,}", " ", tmp)

    return tmp.strip()


def _looks_like_page_number(line: str) -> bool:
    """A standalone integer (up to 4 digits) is almost certainly a
    running page number."""
    return bool(re.fullmatch(r"\d{1,4}", line))


def _write_pages(pages: list[ExtractedPage]) -> None:
    PAGES_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with PAGES_JSONL.open("w", encoding="utf-8") as fh:
        for p in pages:
            fh.write(
                json.dumps(
                    {"pdf_page": p.pdf_page, "raw": p.raw, "cleaned": p.cleaned},
                    ensure_ascii=False,
                )
            )
            fh.write("\n")


__all__ = [
    "ExtractedPage",
    "ExtractionError",
    "PAGES_JSONL",
    "extract_pdf",
    "load_pages",
]
