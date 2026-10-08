"""Implementation of save_guide.

Important: this tool creates a pending draft only. It never writes an
artifact and it cannot mint approval. The separate ``/approve`` route
is the only write path.
"""

from __future__ import annotations

from typing import Any
import unicodedata

from app.ledger import ApprovalRecord, ledger
from app.tools.context import ToolContext
from app.tools.schemas import GuideCitation
from ingest.chapter_guard import enforce_chapter_limit_strict
from ingest.index import lookup_chunk_by_id


def save_guide(
    title: str,
    content: str,
    citations: list[GuideCitation],
    *,
    context: ToolContext,
) -> dict[str, Any]:
    """Validate every cited chunk independently, then create a pending
    operation. No file is created here."""
    if not citations:
        raise ValueError("save_guide requires at least one valid citation")

    citation_dicts: list[dict[str, Any]] = []
    for citation in citations:
        chunk = lookup_chunk_by_id(citation.chunk_id)
        if chunk is None:
            raise ValueError(f"unknown citation chunk_id: {citation.chunk_id}")

        enforce_chapter_limit_strict(
            [chunk],
            context.max_chapter,
            source="tool_save",
        )
        if citation.chapter != chunk.chapter:
            raise ValueError(
                f"citation chapter does not match chunk {citation.chunk_id}"
            )
        if citation.source_filename != chunk.source_filename:
            raise ValueError(
                f"citation filename does not match chunk {citation.chunk_id}"
            )
        if any(page not in chunk.pdf_pages for page in citation.pdf_pages):
            raise ValueError(
                f"citation page is not present on chunk {citation.chunk_id}"
            )
        excerpt = unicodedata.normalize("NFC", citation.excerpt or "")
        if not excerpt.strip():
            raise ValueError(
                f"citation excerpt is empty for chunk {citation.chunk_id}"
            )
        haystack = unicodedata.normalize("NFC", chunk.text)
        if excerpt not in haystack:
            raise ValueError(
                f"citation excerpt is not from chunk {citation.chunk_id}"
            )
        citation_dicts.append(citation.model_dump(mode="json"))

    record: ApprovalRecord = ledger.create_pending(
        title=title,
        content=content,
        citations=citation_dicts,
        chapter_limit=context.max_chapter,
    )
    return {
        "status": "pending_approval",
        "operation_id": record.operation_id,
        "payload_hash": record.payload_hash,
        "draft": record.draft,
        "artifact_created": False,
    }


__all__ = ["save_guide"]
