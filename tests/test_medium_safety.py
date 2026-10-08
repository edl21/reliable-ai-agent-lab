"""Medium-tier safety probes: baseline, mutation, commit token, empty citations."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ledger import (
    ApprovalLedger,
    ApprovalTokenInvalid,
    OperationPayloadTampered,
)
from app.tools.context import ToolContext
from app.tools.save_guide import save_guide
from app.tools.schemas import GuideCitation
from repair.allowlist import validate_source
from repair.loop import BaselineMismatchError, RepairRunner, SUPPLIED_BROKEN_SOURCE
from repair.sandbox import sha256_file


def test_allowlist_rejects_chunk_dict_mutation() -> None:
    src = '''
def select_chunks(chunks, max_chapter, max_words):
    selected = []
    for chunk in chunks:
        chunk["marked"] = True
        selected.append(chunk)
    return selected
'''
    result = validate_source(src)
    assert not result.allowed
    assert "input mutation" in result.reason


def test_allowlist_rejects_chunks_list_mutation_via_alias() -> None:
    src = '''
def select_chunks(chunks, max_chapter, max_words):
    items = chunks
    items.append({"id": "x", "chapter": 1, "score": 1.0, "text": "hi"})
    return items
'''
    result = validate_source(src)
    assert not result.allowed
    assert "input mutation" in result.reason


def test_allowlist_allows_local_selection_structures() -> None:
    src = '''
def select_chunks(chunks, max_chapter, max_words):
    selected = []
    seen = set()
    for chunk in sorted(chunks, key=lambda c: (-c["score"],)):
        if chunk["id"] in seen:
            continue
        selected.append(chunk)
        seen.add(chunk["id"])
    return selected
'''
    result = validate_source(src)
    assert result.allowed, result.reason


def test_baseline_mismatch_rejects_repaired_utility(tmp_path: Path) -> None:
    repair_dir = tmp_path / "repair"
    repair_dir.mkdir()
    # Copy immutable fixtures
    import shutil

    shutil.copy("repair/checks.py", repair_dir / "checks.py")
    shutil.copy("repair/hashes.json", repair_dir / "hashes.json")
    # Write a non-baseline utility
    (repair_dir / "select_chunks.py").write_text(
        "def select_chunks(chunks, max_chapter, max_words):\n    return []\n",
        encoding="utf-8",
    )
    with pytest.raises(BaselineMismatchError):
        RepairRunner(repair_dir=repair_dir, reset_to_supplied=False).run()


def test_reset_to_supplied_matches_hashes_baseline(tmp_path: Path) -> None:
    import json
    import shutil

    repair_dir = tmp_path / "repair"
    repair_dir.mkdir()
    shutil.copy("repair/checks.py", repair_dir / "checks.py")
    shutil.copy("repair/hashes.json", repair_dir / "hashes.json")
    (repair_dir / "select_chunks.py").write_text("broken", encoding="utf-8")
    (repair_dir / "traces").mkdir()
    # Only verify the baseline restore + hash gate, not a full repair.
    runner = RepairRunner(repair_dir=repair_dir, reset_to_supplied=True)
    runner.utility_path.write_text(SUPPLIED_BROKEN_SOURCE, encoding="utf-8")
    expected = json.loads((repair_dir / "hashes.json").read_text())[
        "select_chunks.py_original_sha256"
    ]
    assert sha256_file(runner.utility_path) == expected


def test_direct_commit_rejects_missing_token(tmp_path: Path) -> None:
    ledger = ApprovalLedger(
        artifact_root=tmp_path / "artifacts",
        server_secret="secret",
    )
    record = ledger.create_pending(
        title="t",
        content="c",
        citations=[{"chunk_id": "x"}],
        chapter_limit=1,
    )
    with pytest.raises(ApprovalTokenInvalid):
        ledger._commit(
            operation_id=record.operation_id,
            payload_hash=record.payload_hash,
            approval_token=None,
        )
    assert ledger.write_log.count(record.operation_id) == 0
    assert record.state == "pending"


def test_direct_commit_rejects_mismatched_token(tmp_path: Path) -> None:
    ledger = ApprovalLedger(
        artifact_root=tmp_path / "artifacts",
        server_secret="secret",
    )
    record = ledger.create_pending(
        title="t",
        content="c",
        citations=[{"chunk_id": "x"}],
        chapter_limit=1,
    )
    with pytest.raises(ApprovalTokenInvalid):
        ledger._commit(
            operation_id=record.operation_id,
            payload_hash=record.payload_hash,
            approval_token="0" * 64,
        )
    assert ledger.write_log.count(record.operation_id) == 0


def test_direct_commit_rejects_payload_hash_mismatch(tmp_path: Path) -> None:
    ledger = ApprovalLedger(
        artifact_root=tmp_path / "artifacts",
        server_secret="secret",
    )
    record = ledger.create_pending(
        title="t",
        content="c",
        citations=[{"chunk_id": "x"}],
        chapter_limit=1,
    )
    token = ledger._make_approval_token(record)
    with pytest.raises(OperationPayloadTampered):
        ledger._commit(
            operation_id=record.operation_id,
            payload_hash="0" * 64,
            approval_token=token,
        )
    assert ledger.write_log.count(record.operation_id) == 0


def test_save_guide_rejects_empty_citations() -> None:
    with pytest.raises(ValueError, match="at least one valid citation"):
        save_guide(
            "title",
            "content",
            [],
            context=ToolContext(request_id="t", max_chapter=3),
        )


def test_save_guide_rejects_empty_excerpt() -> None:
    # Use a real chunk when ingest is present; otherwise skip.
    from ingest.chunk import load_chunks

    chunks_path = Path("corpus/chunks.jsonl")
    if not chunks_path.exists():
        pytest.skip("corpus not built")
    chunk = next(c for c in load_chunks() if c.chapter <= 3)
    with pytest.raises(ValueError, match="excerpt is empty"):
        save_guide(
            "title",
            "content",
            [
                GuideCitation(
                    chunk_id=chunk.chunk_id,
                    source_filename=chunk.source_filename,
                    chapter=chunk.chapter,
                    pdf_pages=list(chunk.pdf_pages),
                    excerpt="",
                )
            ],
            context=ToolContext(request_id="t", max_chapter=3),
        )
