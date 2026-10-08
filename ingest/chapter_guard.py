"""Single source of truth for chapter-limit enforcement.

The source assessment says: *"Chapter-limit enforcement must happen in
code, before any model call — including query rewriting, reranking,
direct fetches, summaries, cached answers. No prompt or tool argument
can ever raise the limit."*

This module supplies ONE function, ``enforce_chapter_limit``, called at
every enumerated call site (see plan Phase 1c for the list of ten).
Each call passes ``source=`` so the emitted trace event names the exact
site that enforced the guard — a reviewer can grep a trace and see
which call site rejected an item.

An enforcement test at ``tests/test_chapter_guard_call_sites.py`` walks
the tree and asserts every retrieval-like function names this function
in its source, so a future refactor cannot silently add a fetch/search
without going through the guard.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Protocol, TypeVar

# Recognised call-site tags. Extending this set is fine (add to the list
# in the plan and README), but a call site should always name itself.
CALL_SITES: frozenset[str] = frozenset(
    {
        "index_build",
        "tool_search",
        "tool_fetch",
        "tool_save",
        "answer_retrieval",
        "answer_rerank_in",
        "answer_rerank_out",
        "answer_query_rewrite",
        "answer_summarise_in",
        "answer_summarise_out",
        "answer_cache_read",
        "answer_cache_write",
        "R4_reducer_output",
    }
)


class HasChapter(Protocol):
    """Structural type: any object with a ``chapter`` attribute."""

    chapter: int


T = TypeVar("T")


class OutOfRangeChapter(ValueError):
    """Raised by ``enforce_chapter_limit_strict`` when at least one
    item exceeds ``max_chapter``. The tool layer (search/fetch/save)
    uses the strict variant so an out-of-range chunk_id is rejected
    even if the model guessed a real one."""

    def __init__(self, offending: list[Any], max_chapter: int) -> None:
        self.offending = offending
        self.max_chapter = max_chapter
        super().__init__(
            f"chapter_out_of_range: {len(offending)} item(s) exceed "
            f"max_chapter={max_chapter}"
        )


def enforce_chapter_limit(
    items: Iterable[T],
    max_chapter: int,
    *,
    source: str,
    trace_emit: "TraceEmit | None" = None,
) -> list[T]:
    """Filter ``items`` to those whose chapter is within ``max_chapter``.

    This is the **soft** variant used by retrieval and rerank sites,
    where over-fetching then filtering is the natural pattern. For the
    tool layer (fetch/save) where an out-of-range item is an
    error, use ``enforce_chapter_limit_strict`` instead.

    Args:
        items:        Iterable of chunk-like objects. Each item must
                      expose either ``.chapter`` (attribute) or
                      ``["chapter"]`` (dict-like).
        max_chapter:  Inclusive upper bound. Items with
                      ``chapter <= max_chapter`` pass.
        source:       Call-site tag. MUST be one of ``CALL_SITES`` —
                      unknown tags raise so we can't lose track of
                      where the guard is invoked from.
        trace_emit:   Optional callable ``(kept: int, dropped: int)``
                      the caller uses to append a trace event. Kept
                      generic so this module has zero dep on the
                      trace writer.

    Returns:
        A new list containing only items within the limit. Input is
        never mutated (matches Task 4's "never mutate input" rule and
        keeps the guard usable across retrieval styles)."""
    _validate(max_chapter, source)

    kept: list[T] = []
    dropped = 0
    for item in items:
        chapter = _chapter_of(item)
        if chapter <= max_chapter:
            kept.append(item)
        else:
            dropped += 1

    if trace_emit is not None:
        trace_emit(len(kept), dropped)
    return kept


def enforce_chapter_limit_strict(
    items: Sequence[T],
    max_chapter: int,
    *,
    source: str,
) -> list[T]:
    """Strict variant: raise ``OutOfRangeChapter`` if any item exceeds
    the limit. Used by ``fetch_passage`` and ``save_guide`` so a real
    chunk_id in a later chapter is rejected as an error, not silently
    filtered out.

    Args:
        items:        Sequence of chunk-like objects (list, not
                      generator — we may need to enumerate offenders).
        max_chapter:  Inclusive upper bound.
        source:       Call-site tag.

    Returns:
        The original list of items when all are within range."""
    _validate(max_chapter, source)

    offending: list[T] = []
    for item in items:
        if _chapter_of(item) > max_chapter:
            offending.append(item)
    if offending:
        raise OutOfRangeChapter(offending=offending, max_chapter=max_chapter)
    return list(items)


# --- internals -----------------------------------------------------------


class TraceEmit(Protocol):
    """Callable signature for the optional trace-emit hook."""

    def __call__(self, kept: int, dropped: int) -> None: ...


def _validate(max_chapter: int, source: str) -> None:
    if source not in CALL_SITES:
        raise ValueError(
            f"unknown chapter-guard call site: {source!r}. "
            f"Update ingest.chapter_guard.CALL_SITES and the plan."
        )
    if not isinstance(max_chapter, int) or max_chapter < 1:
        raise ValueError(
            f"max_chapter must be a positive int, got {max_chapter!r}"
        )


def _chapter_of(item: Any) -> int:
    """Extract ``.chapter`` from either an attribute-style or dict-style
    object. Anything else raises ``TypeError`` — a chunk without a
    chapter is a bug in ingestion, not something to paper over."""
    if hasattr(item, "chapter"):
        return int(item.chapter)
    if isinstance(item, dict) and "chapter" in item:
        return int(item["chapter"])
    raise TypeError(
        f"item has no 'chapter' attribute or key: {type(item).__name__}"
    )


__all__ = [
    "CALL_SITES",
    "HasChapter",
    "OutOfRangeChapter",
    "TraceEmit",
    "enforce_chapter_limit",
    "enforce_chapter_limit_strict",
]
