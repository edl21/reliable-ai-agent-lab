"""Unit tests for deterministic claim–citation support scoring."""

from __future__ import annotations

from scripts._claim_support import score_answer, split_claims


def test_split_claims_ignores_tiny_fragments() -> None:
    claims = split_claims("Yes. Ishmael decided to go to sea [abcdef0123456789].")
    assert len(claims) == 1
    assert "Ishmael" in claims[0]


def test_score_answer_flags_unsupported_claim() -> None:
    body = {
        "status": "answered",
        "answer": (
            "Dragons invaded Nantucket overnight [abcdef0123456789]. "
            "Ishmael decided to go to sea [abcdef0123456789]."
        ),
        "citations": [
            {
                "chunk_id": "abcdef0123456789",
                "excerpt": "Ishmael decided to go to sea.",
            }
        ],
    }
    scored = score_answer(body)
    assert scored["claim_count"] == 2
    assert scored["supported_count"] == 1
    assert scored["precision"] == 0.5
    assert scored["unsupported_claims"]


def test_score_answer_insufficient_is_na() -> None:
    scored = score_answer(
        {
            "status": "insufficient_evidence",
            "answer": "No support.",
            "citations": [],
        }
    )
    assert scored["precision"] is None
    assert scored["claim_count"] == 0
