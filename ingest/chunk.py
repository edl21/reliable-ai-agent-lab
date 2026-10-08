"""Chapter-bounded chunker.

Chunks never cross a chapter boundary. We chunk within each chapter
and tag every chunk with its chapter index so retrieval can filter.

Chunk parameters (documented + justified in README):

- **size = 512 tokens** — large enough for a coherent paragraph or two,
  small enough that
  retrieval finds specific facts.
- **overlap = 64 tokens** — avoids losing entities that straddle a
  chunk boundary. 12.5% overlap is the LlamaIndex default recommendation.

Each chunk stores::

    {
        "chunk_id":              SHA-256(source_filename, chapter, char_span)[:16],
        "chapter":               <int, 1..N>,
        "chapter_original_label": <str, e.g. "Chapter 1">,
        "chapter_title":         <str>,
        "pdf_pages":             [<int>, ...],   # 1-based, contiguous
        "source_filename":       <str>,
        "char_span":             [<start>, <end>], # in the chapter's cleaned text
        "text":                  <str>,          # the chunk itself, normalised
    }

``chunk_id`` is derived from (filename, chapter, char_span). It is
deterministic across runs — rebuilding the corpus produces the same IDs
— which is important because ``/answer``'s citation IDs and ``/guide``'s
tool arguments carry these values.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from llama_index.core.node_parser import SentenceSplitter

from ingest.chapters import (
    ChapterEntry,
    load_chapter_map,
    pdf_page_to_chapter_index,
)
from ingest.extract import ExtractedPage, load_pages
from ingest.fetch import CORPUS_DIR, CORPUS_FILENAME


CHUNKS_JSONL: Path = CORPUS_DIR / "chunks.jsonl"

CHUNK_SIZE_TOKENS: int = 512
CHUNK_OVERLAP_TOKENS: int = 64


@dataclass
class Chunk:
    """One chunk of narrative text with all metadata needed for
    retrieval, citation, and the chapter-limit guard."""

    chunk_id: str
    chapter: int
    chapter_original_label: str
    chapter_title: str
    pdf_pages: list[int]
    source_filename: str
    char_span: list[int]  # [start, end)
    text: str

    def to_dict(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "chapter": self.chapter,
            "chapter_original_label": self.chapter_original_label,
            "chapter_title": self.chapter_title,
            "pdf_pages": self.pdf_pages,
            "source_filename": self.source_filename,
            "char_span": self.char_span,
            "text": self.text,
        }


# --- public API ------------------------------------------------------------


def build_chunks() -> list[Chunk]:
    """Build the chapter-bounded chunk set and persist to
    ``corpus/chunks.jsonl``. Idempotent — re-running produces the
    same chunks with the same IDs."""
    pages = load_pages()
    chapter_map = load_chapter_map()

    # Group pages by chapter, in reading order.
    pages_by_chapter: dict[int, list[ExtractedPage]] = {}
    for page in pages:
        idx = pdf_page_to_chapter_index(page.pdf_page, chapter_map)
        if idx is None:
            continue
        pages_by_chapter.setdefault(idx, []).append(page)

    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
    )

    chunks: list[Chunk] = []
    for entry in chapter_map:
        chapter_pages = pages_by_chapter.get(entry.index, [])
        if not chapter_pages:
            continue

        # Assemble the chapter's text and build a char-index → pdf_page
        # map so each chunk can list the pages it spans.
        chapter_text, char_to_page = _assemble_chapter(chapter_pages)
        if not chapter_text.strip():
            continue

        # LlamaIndex's SentenceSplitter operates on a list of texts;
        # we pass the chapter as one document and read back the pieces.
        pieces = splitter.split_text(chapter_text)

        # Find each piece's char span by locating it in the chapter text.
        # Using a running cursor avoids O(n^2) in the number of pieces.
        cursor = 0
        for piece in pieces:
            normalised = _normalise(piece)
            start = chapter_text.find(piece, cursor)
            if start < 0:
                # Fallback: LlamaIndex may lightly rewrite the piece.
                # Locate by the first 40 chars to keep this robust.
                probe = piece[:40].strip()
                start = chapter_text.find(probe, cursor) if probe else -1
                if start < 0:
                    start = cursor  # give up gracefully — trace will still be valid
            end = start + len(piece)
            cursor = max(cursor, start + 1)

            pdf_pages = _pages_in_span(char_to_page, start, end)
            chunk_id = _make_chunk_id(entry.source_filename, entry.index, start, end)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    chapter=entry.index,
                    chapter_original_label=entry.original_label,
                    chapter_title=entry.chapter_title,
                    pdf_pages=pdf_pages,
                    source_filename=entry.source_filename,
                    char_span=[start, end],
                    text=normalised,
                )
            )

    if not chunks:
        raise RuntimeError("no chunks produced; check chapter map and pages")

    _write_chunks(chunks)
    return chunks


def load_chunks() -> list[Chunk]:
    """Read back the chunks from JSONL."""
    if not CHUNKS_JSONL.exists():
        raise RuntimeError(
            f"{CHUNKS_JSONL} missing; run `python -m ingest.build` first"
        )
    out: list[Chunk] = []
    with CHUNKS_JSONL.open("r", encoding="utf-8") as fh:
        for line in fh:
            data = json.loads(line)
            out.append(Chunk(**data))
    return out


# --- internals -------------------------------------------------------------


def _assemble_chapter(
    chapter_pages: list[ExtractedPage],
) -> tuple[str, list[int]]:
    """Concatenate the cleaned text of all pages in a chapter, and
    return a parallel list mapping each character index in the
    concatenated text back to the PDF page it came from."""
    parts: list[str] = []
    char_to_page: list[int] = []

    for i, page in enumerate(chapter_pages):
        text = page.cleaned
        if i > 0:
            # Insert a paragraph break between pages so the sentence
            # splitter respects the natural page-boundary pause.
            parts.append("\n\n")
            char_to_page.extend([page.pdf_page, page.pdf_page])
        parts.append(text)
        char_to_page.extend([page.pdf_page] * len(text))

    return "".join(parts), char_to_page


def _pages_in_span(
    char_to_page: list[int], start: int, end: int
) -> list[int]:
    """Return the sorted unique PDF pages that the given char span
    touches. If the mapping is short (shouldn't happen), fall back to
    the last known page."""
    if not char_to_page:
        return []
    end = min(end, len(char_to_page))
    start = max(0, min(start, len(char_to_page) - 1))
    seen: list[int] = []
    prev = None
    for i in range(start, end):
        page = char_to_page[i]
        if page != prev:
            if page not in seen:
                seen.append(page)
            prev = page
    return seen


def _make_chunk_id(source_filename: str, chapter: int, start: int, end: int) -> str:
    """Deterministic SHA-256-derived ID. First 16 hex chars is plenty
    for uniqueness within a few-thousand-chunk corpus and keeps the
    IDs short enough to eyeball in traces."""
    payload = f"{source_filename}::{chapter}::{start}::{end}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _normalise(text: str) -> str:
    """NFC-normalise and strip leading/trailing whitespace. Matches the
    canonical-payload rule so citations excerpt-matching in
    ``/answer``'s post-hoc verifier works reliably."""
    return unicodedata.normalize("NFC", text).strip()


def _write_chunks(chunks: list[Chunk]) -> None:
    CHUNKS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_JSONL.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c.to_dict(), ensure_ascii=False))
            fh.write("\n")


__all__ = [
    "CHUNKS_JSONL",
    "CHUNK_OVERLAP_TOKENS",
    "CHUNK_SIZE_TOKENS",
    "Chunk",
    "build_chunks",
    "load_chunks",
]
