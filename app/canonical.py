"""Canonical JSON serialisation and SHA-256 hashing — one function used
everywhere that needs a deterministic content fingerprint.

Referenced from:

- ``app/routes/guide.py``    — hashes the pending draft payload.
- ``app/routes/approve.py``  — re-hashes on ``/approve`` for the
                                tamper-check (source: "reusing an
                                operation_id with different content must
                                be rejected").
- ``app/ledger.py``          — dedup / lookup key.
- ``adapter/faults.py``      — R3 replay looks up the ledger entry by
                                (operation_id, payload_hash).

Determinism rules (all pinned; a divergence anywhere would break every
idempotency and R3 assertion):

1. **Sorted keys.** ``json.dumps(sort_keys=True)``.
2. **No whitespace.** ``separators=(",", ":")``.
3. **UTF-8 encoding.** ``.encode("utf-8")`` at the very end.
4. **Unicode NFC normalisation** on every string, at every depth, so
   inputs that differ only in composition form (NFC vs. NFD) hash to
   the same bytes.
5. **Floats are disallowed** inside canonical payloads. The
   ``guide+citations+chapter_limit`` schema uses only ``int`` and
   ``str``. A float in the input tree raises ``TypeError`` — we would
   rather fail loud than hash a lossy repr.
6. **Preserve list order.** Lists (notably the citations array) keep
   their input order. Callers are responsible for sorting the citations
   list before hashing if a canonical order is required — the current
   contract is "first-appearance-in-answer" order.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from typing import Any


class CanonicalisationError(TypeError):
    """Raised when the input tree contains a value we refuse to hash
    (currently: ``float``, ``bytes``, sets, or arbitrary objects)."""


def canonical_json_bytes(obj: Any) -> bytes:
    """Serialise ``obj`` to the canonical JSON byte representation.

    See module docstring for the rules. Returns UTF-8 bytes ready for
    hashing or wire transmission."""
    normalised = _normalise(obj)
    return json.dumps(
        normalised,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    """Hex-encoded SHA-256 of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def canonical_hash(obj: Any) -> str:
    """Convenience: canonical-serialise then SHA-256 in one call."""
    return sha256_hex(canonical_json_bytes(obj))


def _normalise(value: Any) -> Any:
    """Recursively coerce a value into a canonicalisable JSON tree.

    - dicts: keys must be strings; NFC-normalise the keys and recurse
      on values.
    - lists/tuples: recurse (tuples become lists).
    - strings: NFC-normalise.
    - int, bool, None: passthrough.
    - float: reject (rule 5).
    - anything else: reject (rule 5).
    """
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if value is None or isinstance(value, bool):
        # ``bool`` must be checked before ``int`` because bool is a subclass.
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        raise CanonicalisationError(
            "floats are not allowed in canonical payloads; use int or str"
        )
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise CanonicalisationError(
                    f"canonical JSON keys must be strings, got {type(k).__name__}"
                )
            out[unicodedata.normalize("NFC", k)] = _normalise(v)
        return out
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    raise CanonicalisationError(
        f"cannot canonicalise value of type {type(value).__name__}"
    )


__all__ = [
    "CanonicalisationError",
    "canonical_hash",
    "canonical_json_bytes",
    "sha256_hex",
]
