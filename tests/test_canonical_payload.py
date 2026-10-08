"""Canonical-payload determinism tests.

The adversarial review's #1 risk is that ``/guide`` and ``/approve``
compute different hashes for what a human would call "the same
payload" — breaking every idempotency assertion and the R3 replay
guarantee. This test locks the rules pinned in ``app/canonical.py``:

- Sorted keys.
- No whitespace.
- Unicode NFC normalisation at every string depth.
- Floats disallowed.
- List order preserved.
"""

from __future__ import annotations

import unicodedata

import pytest

from app.canonical import (
    CanonicalisationError,
    canonical_hash,
    canonical_json_bytes,
    sha256_hex,
)


def test_key_reordering_produces_identical_hash() -> None:
    a = {"z": 1, "a": 2, "m": 3}
    b = {"a": 2, "m": 3, "z": 1}
    assert canonical_hash(a) == canonical_hash(b)


def test_nested_key_reordering() -> None:
    a = {"outer": {"z": 1, "a": [{"y": 2, "x": 1}]}}
    b = {"outer": {"a": [{"x": 1, "y": 2}], "z": 1}}
    assert canonical_hash(a) == canonical_hash(b)


def test_nfc_vs_nfd_hash_the_same() -> None:
    # "café" in composed (NFC) vs. decomposed (NFD) form.
    composed = "caf\u00e9"                       # café — one code point
    decomposed = "cafe\u0301"                    # café — e + combining acute
    assert composed != decomposed
    assert unicodedata.normalize("NFC", composed) == unicodedata.normalize(
        "NFC", decomposed
    )

    a = {"title": composed}
    b = {"title": decomposed}
    assert canonical_hash(a) == canonical_hash(b)


def test_nfc_normalisation_applied_to_keys_too() -> None:
    a = {"caf\u00e9": 1}
    b = {"cafe\u0301": 1}
    assert canonical_hash(a) == canonical_hash(b)


def test_whitespace_in_input_does_not_affect_hash() -> None:
    # We control the whitespace in the OUTPUT via separators; the
    # input's own key ordering / dict construction is irrelevant.
    import json

    equivalent_a = json.loads('{"a": 1, "b": [1, 2, 3]}')
    equivalent_b = json.loads('{"b":[1,2,3],"a":1}')
    assert canonical_hash(equivalent_a) == canonical_hash(equivalent_b)


def test_list_order_is_preserved() -> None:
    # Lists are ordered; two different orders must NOT hash the same,
    # even though {a,b,c} == {c,b,a} as sets. This matches the source's
    # citations-in-first-appearance-order rule.
    a = {"citations": ["chunk_a", "chunk_b", "chunk_c"]}
    b = {"citations": ["chunk_c", "chunk_b", "chunk_a"]}
    assert canonical_hash(a) != canonical_hash(b)


def test_floats_are_rejected() -> None:
    with pytest.raises(CanonicalisationError):
        canonical_json_bytes({"score": 0.87})


def test_nested_floats_are_rejected() -> None:
    with pytest.raises(CanonicalisationError):
        canonical_json_bytes({"citations": [{"score": 1.0}]})


def test_sets_are_rejected() -> None:
    with pytest.raises(CanonicalisationError):
        canonical_json_bytes({"tags": {"a", "b"}})


def test_bytes_are_rejected() -> None:
    with pytest.raises(CanonicalisationError):
        canonical_json_bytes({"blob": b"raw"})


def test_bool_is_not_int() -> None:
    """Guard against Python's ``bool ⊂ int`` — ``True`` must serialise
    as ``true``, not ``1``."""
    out = canonical_json_bytes({"ok": True, "count": 1})
    assert out == b'{"count":1,"ok":true}'


def test_output_shape_is_stable() -> None:
    """Lock the exact byte string for a known input so a regression in
    serialisation is impossible to miss."""
    obj = {
        "chapter_limit": 5,
        "title": "The Ring",
        "content": "a b c",
        "citations": [{"chunk_id": "x", "chapter": 1}],
    }
    expected = (
        b'{"chapter_limit":5,'
        b'"citations":[{"chapter":1,"chunk_id":"x"}],'
        b'"content":"a b c",'
        b'"title":"The Ring"}'
    )
    assert canonical_json_bytes(obj) == expected


def test_sha256_hex_matches_manual_computation() -> None:
    import hashlib

    obj = {"a": 1}
    manual = hashlib.sha256(b'{"a":1}').hexdigest()
    assert sha256_hex(canonical_json_bytes(obj)) == manual
    assert canonical_hash(obj) == manual
