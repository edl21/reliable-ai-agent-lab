"""Deterministic claim–citation support scoring for live answer gates.

This is intentionally not an LLM judge. It splits answered prose into
substantive sentences and checks whether each cited sentence's nearby
excerpt shares enough content tokens with the claim.
"""

from __future__ import annotations

import re
from typing import Any


_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "is",
    "are",
    "was",
    "were",
    "be",
    "by",
    "with",
    "as",
    "at",
    "from",
    "that",
    "this",
    "it",
    "he",
    "she",
    "they",
    "his",
    "her",
    "their",
}


def split_claims(answer: str) -> list[str]:
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


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]{2,}", (text or "").lower())
    return {w for w in words if w not in _STOPWORDS}


def claim_supported(claim: str, citations: list[dict[str, Any]]) -> bool:
    inline = re.findall(r"\[([a-f0-9]{16})\]", claim)
    if not inline:
        return False
    claim_tokens = _tokens(re.sub(r"\[[a-f0-9]{16}\]", " ", claim))
    if not claim_tokens:
        return True
    by_id = {c.get("chunk_id"): c for c in citations if isinstance(c, dict)}
    for chunk_id in inline:
        cit = by_id.get(chunk_id)
        if not cit:
            continue
        excerpt_tokens = _tokens(str(cit.get("excerpt") or ""))
        if not excerpt_tokens:
            continue
        overlap = claim_tokens & excerpt_tokens
        # Require modest lexical overlap, or accept if excerpt is long and
        # the claim mostly restates cited content.
        if len(overlap) >= 2 or (
            len(overlap) >= 1 and len(excerpt_tokens) >= 8
        ):
            return True
    return False


def score_answer(body: dict[str, Any]) -> dict[str, Any]:
    """Return supported/total claim-citation precision fields."""
    status = body.get("status")
    answer = body.get("answer") or ""
    citations = body.get("citations") or []
    if status != "answered":
        return {
            "status": status,
            "claim_count": 0,
            "supported_count": 0,
            "precision": None,
            "unsupported_claims": [],
        }
    claims = split_claims(answer)
    unsupported = [c for c in claims if not claim_supported(c, citations)]
    supported = len(claims) - len(unsupported)
    precision = (supported / len(claims)) if claims else None
    return {
        "status": status,
        "claim_count": len(claims),
        "supported_count": supported,
        "precision": precision,
        "unsupported_claims": unsupported[:5],
    }


__all__ = ["claim_supported", "score_answer", "split_claims"]
