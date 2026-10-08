"""Live /guide batch gate: two fixed goals must reach pending_approval."""

from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fastapi.testclient import TestClient

import adapter.model as model_mod
from adapter import runtime
from app.main import app
from ingest.index import lookup_chunk_by_id


GOALS = [
    (
        "live_guide_ishmael",
        "Create a short cited reading guide explaining Ishmael's early situation "
        "across two permitted chapters.",
        2,
    ),
    (
        "live_guide_queequeg",
        "Create a short cited reading guide about Ishmael meeting Queequeg and its "
        "consequences within the permitted chapters.",
        5,
    ),
]


def _citations_valid(draft: dict) -> tuple[bool, list[str]]:
    errors: list[str] = []
    citations = draft.get("citations") or []
    if not citations:
        return False, ["no_citations"]
    for cit in citations:
        chunk = lookup_chunk_by_id(cit.get("chunk_id") or "")
        if chunk is None:
            errors.append("unknown_chunk")
            continue
        excerpt = unicodedata.normalize("NFC", cit.get("excerpt") or "")
        haystack = unicodedata.normalize("NFC", chunk.text)
        if not excerpt.strip() or excerpt not in haystack:
            errors.append("excerpt_not_verbatim")
        if cit.get("chapter") != chunk.chapter:
            errors.append("chapter_mismatch")
        if cit.get("source_filename") != chunk.source_filename:
            errors.append("filename_mismatch")
    return (not errors), errors


def main() -> int:
    model_mod._SINGLETON = None
    runtime.clear_override()
    client = TestClient(app, raise_server_exceptions=False)

    results = []
    for rid, goal, max_chapter in GOALS:
        resp = client.post(
            "/guide",
            params={"trace_check_name": "live_guide"},
            json={
                "request_id": rid,
                "goal": goal,
                "max_chapter": max_chapter,
            },
        )
        body = (
            resp.json()
            if resp.headers.get("content-type", "").startswith("application/json")
            else {"raw": resp.text}
        )
        draft = body.get("draft") if isinstance(body, dict) else None
        citations_ok, cite_errors = (
            _citations_valid(draft) if isinstance(draft, dict) else (False, ["no_draft"])
        )
        approved = False
        artifact_id = None
        if (
            isinstance(body, dict)
            and body.get("status") == "pending_approval"
            and body.get("operation_id")
            and citations_ok
        ):
            approve = client.post(
                "/approve",
                json={
                    "operation_id": body["operation_id"],
                    "approve": True,
                },
            )
            approve_body = (
                approve.json()
                if approve.headers.get("content-type", "").startswith(
                    "application/json"
                )
                else {}
            )
            approved = (
                approve.status_code == 200
                and approve_body.get("status") == "saved"
            )
            artifact_id = approve_body.get("artifact_id")

        row = {
            "id": rid,
            "status_code": resp.status_code,
            "status": body.get("status") if isinstance(body, dict) else None,
            "operation_id": body.get("operation_id") if isinstance(body, dict) else None,
            "stubbed": (body.get("metrics") or {}).get("stubbed")
            if isinstance(body, dict)
            else None,
            "rejected_save_attempts": (body.get("metrics") or {}).get(
                "rejected_save_attempts"
            )
            if isinstance(body, dict)
            else None,
            "citations_ok": citations_ok,
            "citation_errors": cite_errors,
            "approved": approved,
            "artifact_id": artifact_id,
            "error": body.get("error") if isinstance(body, dict) else None,
        }
        results.append(row)
        print(json.dumps(row, indent=2))

        # Keep local live traces out of git; just ensure they exist for review.
        trace = Path(f"traces/task2/live_guide_{rid}.jsonl")
        print("trace_exists", trace.exists())

    gate = all(
        r["status_code"] == 200
        and r["status"] == "pending_approval"
        and r["citations_ok"]
        and r["approved"]
        and r["stubbed"] is False
        for r in results
    )
    print("--- GATE ---")
    print("live_guide_gate", gate)
    print("count", len(results))
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
