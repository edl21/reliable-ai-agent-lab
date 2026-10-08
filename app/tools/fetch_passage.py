"""Implementation of the model-callable fetch_passage tool."""

from __future__ import annotations

from typing import Any

from app.tools.context import ToolContext, get_test_untrusted_suffix
from ingest.chapter_guard import (
    OutOfRangeChapter,
    enforce_chapter_limit_strict,
)
from ingest.index import lookup_chunk_by_id


def fetch_passage(
    chunk_id: str,
    *,
    context: ToolContext,
) -> dict[str, Any]:
    """Fetch one chunk and independently reject an out-of-range ID."""
    chunk = lookup_chunk_by_id(chunk_id)
    if chunk is None:
        raise ValueError(f"unknown chunk_id: {chunk_id}")

    # Strict guard is intentional: fetch must reject a real later-chapter
    # ID, not silently return an empty result.
    enforce_chapter_limit_strict(
        [chunk],
        context.max_chapter,
        source="tool_fetch",
    )

    content = chunk.text
    suffix = get_test_untrusted_suffix()
    if suffix:
        content += suffix

    return {
        "chunk_id": chunk.chunk_id,
        "chapter": chunk.chapter,
        "chapter_original_label": chunk.chapter_original_label,
        "chapter_title": chunk.chapter_title,
        "source_filename": chunk.source_filename,
        "pdf_pages": chunk.pdf_pages,
        "content": content,
    }


__all__ = ["fetch_passage", "OutOfRangeChapter"]
