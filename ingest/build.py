"""One-command build: fetch, extract, chapter map, chunk, index.

Usage::

    python -m ingest.build

Idempotent. Prints a summary at each step so users can follow
along. Any exception halts the pipeline and returns a non-zero exit.
"""

from __future__ import annotations

import sys

from ingest.chapters import build_chapter_map
from ingest.chunk import build_chunks
from ingest.extract import extract_pdf
from ingest.fetch import fetch_corpus
from ingest.index import build_index


def main() -> int:
    print("Reliable AI Agent Lab — corpus build")
    print("=" * 60)

    print("[1/5] Fetching corpus...")
    manifest = fetch_corpus()
    print(
        f"      OK: {manifest.title} — {manifest.file_size_bytes:,} bytes, "
        f"SHA-256={manifest.sha256[:12]}..."
    )

    print("[2/5] Extracting PDF text...")
    pages = extract_pdf()
    total_words = sum(len(p.cleaned.split()) for p in pages)
    print(
        f"      OK: {len(pages)} pages survived cleanup; "
        f"{total_words:,} words after cleaning"
    )

    print("[3/5] Building chapter map...")
    chapter_map = build_chapter_map(pages=pages)
    print(
        f"      OK: {len(chapter_map)} chapters (1={chapter_map[0].original_label!r}, "
        f"{len(chapter_map)}={chapter_map[-1].original_label!r})"
    )

    print("[4/5] Chunking (chapter-bounded)...")
    chunks = build_chunks()
    print(f"      OK: {len(chunks)} chunks written")

    print("[5/5] Building Chroma index...")
    count = build_index(chunks=chunks)
    print(f"      OK: {count} chunks embedded and indexed")

    print()
    print("Done. Corpus ready for /answer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
