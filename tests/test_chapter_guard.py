"""Behaviour tests for the chapter-limit guard.

The grep-based coverage test elsewhere ensures every retrieval-like
function calls the guard. THIS test locks the guard's behaviour itself:

- Soft filter keeps items ``chapter <= max_chapter``, drops the rest,
  never mutates the input.
- Strict variant raises ``OutOfRangeChapter`` when any item exceeds.
- Unknown call-site tag is refused (fails loud rather than silently
  accepting the wrong site).
- Non-positive ``max_chapter`` is refused (a source-required guard on
  the guard).
- Works on both attribute-style and dict-style items.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ingest.chapter_guard import (
    OutOfRangeChapter,
    enforce_chapter_limit,
    enforce_chapter_limit_strict,
)


@dataclass
class _Chunk:
    chunk_id: str
    chapter: int


def test_soft_filter_keeps_within_range_drops_above() -> None:
    items = [
        _Chunk("a", 1),
        _Chunk("b", 3),
        _Chunk("c", 5),
        _Chunk("d", 7),
    ]
    kept = enforce_chapter_limit(items, 5, source="answer_retrieval")
    assert [c.chunk_id for c in kept] == ["a", "b", "c"]


def test_soft_filter_is_inclusive_of_max_chapter() -> None:
    # Source: chapter limit is INCLUSIVE. max_chapter=5 keeps chapter=5.
    items = [_Chunk("x", 5)]
    kept = enforce_chapter_limit(items, 5, source="answer_retrieval")
    assert kept == items


def test_soft_filter_does_not_mutate_input() -> None:
    items = [_Chunk("a", 1), _Chunk("b", 10)]
    snapshot = list(items)
    enforce_chapter_limit(items, 5, source="answer_retrieval")
    assert items == snapshot


def test_soft_filter_works_on_dict_items() -> None:
    items = [
        {"chunk_id": "a", "chapter": 1},
        {"chunk_id": "b", "chapter": 6},
    ]
    kept = enforce_chapter_limit(items, 5, source="answer_retrieval")
    assert kept == [{"chunk_id": "a", "chapter": 1}]


def test_strict_variant_raises_on_out_of_range() -> None:
    items = [_Chunk("a", 1), _Chunk("b", 10)]
    with pytest.raises(OutOfRangeChapter) as exc:
        enforce_chapter_limit_strict(items, 5, source="tool_fetch")
    assert exc.value.max_chapter == 5
    assert len(exc.value.offending) == 1


def test_strict_variant_passes_when_all_in_range() -> None:
    items = [_Chunk("a", 1), _Chunk("b", 3)]
    out = enforce_chapter_limit_strict(items, 5, source="tool_fetch")
    assert [c.chunk_id for c in out] == ["a", "b"]


def test_unknown_source_tag_raises() -> None:
    with pytest.raises(ValueError, match="unknown chapter-guard call site"):
        enforce_chapter_limit([], 5, source="nonexistent_site")


def test_non_positive_max_chapter_raises() -> None:
    with pytest.raises(ValueError, match="max_chapter must be a positive int"):
        enforce_chapter_limit([], 0, source="answer_retrieval")
    with pytest.raises(ValueError, match="max_chapter must be a positive int"):
        enforce_chapter_limit([], -1, source="answer_retrieval")


def test_trace_emit_hook_receives_kept_and_dropped_counts() -> None:
    calls: list[tuple[int, int]] = []

    def emit(kept: int, dropped: int) -> None:
        calls.append((kept, dropped))

    items = [_Chunk("a", 1), _Chunk("b", 6), _Chunk("c", 7)]
    enforce_chapter_limit(items, 5, source="answer_retrieval", trace_emit=emit)
    assert calls == [(1, 2)]


def test_item_without_chapter_raises_typeerror() -> None:
    class NoChapter:
        pass

    with pytest.raises(TypeError, match="no 'chapter' attribute or key"):
        enforce_chapter_limit([NoChapter()], 5, source="answer_retrieval")
