"""Unit tests for ingest modules that DON'T need network / built corpus.

The heavy end-to-end tests live in ``test_task1_endpoint.py`` (skip
when the corpus isn't built)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ingest.chapters import _to_int, _trim_title  # type: ignore[attr-defined]


# --- _trim_title tests -----------------------------------------------


def test_trim_title_stops_at_first_lower_case_word() -> None:
    out = _trim_title("THE WHITENESS OF THE WHALE Ishmael considers")
    assert out == "THE WHITENESS OF THE WHALE"


def test_trim_title_handles_apostrophes() -> None:
    out = _trim_title("AHAB'S BOAT AND CREW 'Now we had better")
    assert out == "AHAB'S BOAT AND CREW"


def test_trim_title_empty_input() -> None:
    assert _trim_title("") == ""


def test_trim_title_all_uppercase_stays() -> None:
    out = _trim_title("THE CHASE")
    assert out == "THE CHASE"


def test_trim_title_preserves_title_case() -> None:
    assert _trim_title("Loomings") == "Loomings"


# --- _to_int (roman + arabic) tests ---------------------------------------


def test_to_int_arabic() -> None:
    assert _to_int("1") == 1
    assert _to_int("12") == 12


def test_to_int_roman() -> None:
    assert _to_int("I") == 1
    assert _to_int("IV") == 4
    assert _to_int("V") == 5
    assert _to_int("IX") == 9
    assert _to_int("X") == 10
    assert _to_int("XII") == 12
    assert _to_int("XIII") == 13


def test_to_int_lowercase_roman() -> None:
    assert _to_int("i") == 1
    assert _to_int("iv") == 4


# --- corpus-artefact tests (skip when unbuilt) ----------------------------


_MANIFEST = Path("corpus/manifest.json")
_CHAPTER_MAP = Path("corpus/chapter_map.json")
_CHUNKS = Path("corpus/chunks.jsonl")

needs_corpus = pytest.mark.skipif(
    not all(p.exists() for p in (_MANIFEST, _CHAPTER_MAP, _CHUNKS)),
    reason="corpus not built; run `python -m ingest.build`",
)


@needs_corpus
def test_manifest_has_required_fields() -> None:
    data = json.loads(_MANIFEST.read_text())
    # Source-pinned fingerprint fields.
    for field in (
        "title",
        "author",
        "source_url",
        "download_date_utc",
        "sha256",
        "file_size_bytes",
        "rights",
        "rights_url",
        "pdf_pages",
        "cleaned_word_count",
    ):
        assert field in data, f"manifest missing {field}"
    # SHA-256 is a 64-char hex.
    assert len(data["sha256"]) == 64
    # Word count matches source expectation (>= 150k).
    assert data["cleaned_word_count"] >= 150_000


@needs_corpus
def test_chapter_map_has_at_least_15_chapters() -> None:
    m = json.loads(_CHAPTER_MAP.read_text())
    assert len(m) >= 15, "source requires >= 15 chapters"


@needs_corpus
def test_chapter_map_indices_are_1_to_n_dense() -> None:
    m = json.loads(_CHAPTER_MAP.read_text())
    indices = [e["index"] for e in m]
    assert indices == list(range(1, len(m) + 1))


@needs_corpus
def test_chapter_map_starts_with_first_chapter() -> None:
    m = json.loads(_CHAPTER_MAP.read_text())
    assert m[0]["original_label"] == "Chapter 1"


@needs_corpus
def test_chunks_never_cross_chapter_boundaries() -> None:
    """Every chunk carries exactly ONE chapter integer."""
    with _CHUNKS.open("r") as f:
        for line in f:
            c = json.loads(line)
            assert isinstance(c["chapter"], int)
            assert c["chapter"] >= 1


@needs_corpus
def test_chunk_ids_are_stable_and_unique() -> None:
    ids: list[str] = []
    with _CHUNKS.open("r") as f:
        for line in f:
            c = json.loads(line)
            ids.append(c["chunk_id"])
            # 16-hex-char ID.
            assert len(c["chunk_id"]) == 16
    # No duplicates.
    assert len(ids) == len(set(ids)), "duplicate chunk_ids found"
