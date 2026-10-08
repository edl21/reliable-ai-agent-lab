"""Live /answer batch for Critical tier gate. Run with gateway env loaded.

Fails the gate when any answered response lacks citations/inline IDs or
when a claim-citation support score falls below threshold.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fastapi.testclient import TestClient

import adapter.model as model_mod
from adapter import runtime
from app.main import app


def _split_claims(answer: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", (answer or "").strip())
    claims: list[str] = []
    for part in parts:
        text = part.strip()
        if not text:
            continue
        words = re.findall(r"[A-Za-z]{2,}", re.sub(r"\[[a-f0-9]{16}\]", " ", text))
        if len(words) >= 3:
            claims.append(text)
    return claims


def _claim_supported(claim: str, citations: list[dict]) -> bool:
    """Deterministic floor: claim must cite an id whose excerpt overlaps tokens."""
    ids = re.findall(r"\[([a-f0-9]{16})\]", claim)
    if not ids:
        return False
    by_id = {c.get("chunk_id"): c for c in citations if c.get("chunk_id")}
    claim_tokens = {
        t.lower()
        for t in re.findall(r"[A-Za-z]{3,}", re.sub(r"\[[a-f0-9]{16}\]", " ", claim))
    }
    for cid in ids:
        cit = by_id.get(cid)
        if not cit:
            continue
        excerpt = (cit.get("excerpt") or "").strip()
        if len(excerpt) < 20:
            continue
        excerpt_tokens = {t.lower() for t in re.findall(r"[A-Za-z]{3,}", excerpt)}
        if claim_tokens & excerpt_tokens:
            return True
    return False


def _precision(answer: str, citations: list[dict]) -> float | None:
    claims = _split_claims(answer)
    if not claims:
        return None
    supported = sum(1 for c in claims if _claim_supported(c, citations))
    return supported / len(claims)


def main() -> int:
    model_mod._SINGLETON = None
    runtime.clear_override()
    client = TestClient(app, raise_server_exceptions=False)

    questions = [
        ("live_crit_ishmael", "Who is Ishmael and why does he go to sea?", 2),
        ("live_crit_queequeg", "Who is Queequeg and how does Ishmael meet him?", 5),
        ("live_crit_sermon", "What lesson does Father Mapple draw from Jonah?", 10),
    ]
    results = []
    for rid, q, ch in questions:
        resp = client.post(
            "/answer",
            params={"trace_check_name": "live_critical"},
            json={"request_id": rid, "question": q, "max_chapter": ch},
        )
        body = (
            resp.json()
            if resp.headers.get("content-type", "").startswith("application/json")
            else {"raw": resp.text}
        )
        trace = Path(f"traces/task1/live_critical_{rid}.jsonl")
        events = (
            [json.loads(line) for line in trace.read_text().splitlines()]
            if trace.exists()
            else []
        )
        lingering = False
        unresolved_kinds: list[str] = []
        precision: float | None = None
        if isinstance(body, dict) and body.get("status") == "answered":
            citations = body.get("citations") or []
            answer_text = body.get("answer") or ""
            if not citations:
                lingering = True
                unresolved_kinds.append("answered_without_citations")
            if not re.search(r"\[[a-f0-9]{16}\]", answer_text):
                lingering = True
                unresolved_kinds.append("answered_without_inline_citations")
            precision = _precision(answer_text, citations)
            if precision is not None and precision < 1.0:
                lingering = True
                unresolved_kinds.append("unsupported_claim_citation_pairs")
            oks = [
                e
                for e in events
                if e.get("operation") == "validation" and e.get("outcome") == "ok"
            ]
            if not oks:
                lingering = True
            for cit in citations:
                if not cit.get("excerpt") or not cit.get("source_filename"):
                    lingering = True
                    unresolved_kinds.append("empty_citation_fields")
            for e in events:
                if (
                    e.get("operation") == "validation"
                    and e.get("outcome") == "error"
                    and not (e.get("notes") or {}).get("will_retry")
                ):
                    lingering = True
                    for err in (e.get("notes") or {}).get("verifier_errors") or []:
                        unresolved_kinds.append(err.get("kind"))
        row = {
            "id": rid,
            "status_code": resp.status_code,
            "status": body.get("status") if isinstance(body, dict) else None,
            "n_citations": len(body.get("citations") or []) if isinstance(body, dict) else 0,
            "stubbed": (body.get("metrics") or {}).get("stubbed")
            if isinstance(body, dict)
            else None,
            "citation_precision": precision,
            "lingering_verifier_errors": lingering,
            "unresolved_kinds": unresolved_kinds,
            "answer_head": (
                (body.get("answer") or "")[:200]
                if isinstance(body, dict)
                else resp.text[:200]
            ),
            "detail": body.get("detail") if isinstance(body, dict) else None,
        }
        results.append(row)
        print(json.dumps(row, indent=2))

    gate = all(
        r["status_code"] == 200 and not r["lingering_verifier_errors"] for r in results
    )
    all_live = all(r["stubbed"] is False for r in results)
    print("--- GATE ---")
    print("live_batch_gate", gate)
    print("all_live", all_live)
    return 0 if gate and all_live else 1


if __name__ == "__main__":
    raise SystemExit(main())
