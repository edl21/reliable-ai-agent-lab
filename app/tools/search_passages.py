"""Implementation of the model-callable search_passages tool."""

from __future__ import annotations

from typing import Any

from app.tools.context import ToolContext
from ingest.chapter_guard import enforce_chapter_limit
from ingest.index import retrieve


def search_passages(
    query: str,
    *,
    context: ToolContext,
) -> dict[str, Any]:
    """Search and independently enforce the request chapter limit."""
    candidates = retrieve(query, context.max_chapter, top_k=8)
    guarded = enforce_chapter_limit(
        candidates,
        context.max_chapter,
        source="tool_search",
    )
    return {
        "results": [
            {
                "chunk_id": chunk.chunk_id,
                "chapter": chunk.chapter,
                "source_filename": chunk.source_filename,
                "pdf_pages": chunk.pdf_pages,
                "excerpt": chunk.text[:500],
            }
            for chunk in guarded
        ]
    }


__all__ = ["search_passages"]
