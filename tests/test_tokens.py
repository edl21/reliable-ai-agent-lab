"""Token-counting invariants.

The R4 injector and the app use the SAME functions here; a divergence
would break the R4 assertion. These tests lock the interface so a
future refactor cannot silently change behaviour.

Note: tiktoken lazy-downloads its BPE table on first use. Tests that
actually count tokens are skipped if the network fetch fails (e.g. in
a sandboxed CI environment); the constants and interface shape tests
still run so we catch a refactor that changes them offline."""

from __future__ import annotations

import pytest

from adapter.tokens import (
    EVIDENCE_BUDGET,
    TOKENS_PER_MESSAGE,
    TOKENS_PER_REPLY_PRIMER,
    TOTAL_CONTEXT_LIMIT_DEFAULT,
    count_messages,
    count_output_reservation,
    count_text,
    count_tool_schemas,
    encoding_name,
)


def _tiktoken_available() -> bool:
    """True iff the BPE table can be loaded (cached or downloadable).

    Some sandboxed environments (including the one we build in) cannot
    reach ``openaipublic.blob.core.windows.net`` — tiktoken then raises
    a ``ProxyError`` or ``ConnectionError`` on first ``encode`` call.
    We surface that as a skip so the rest of the suite still runs."""
    try:
        count_text("probe")
    except Exception:  # noqa: BLE001 — any load-time error means unavailable
        return False
    return True


needs_tiktoken = pytest.mark.skipif(
    not _tiktoken_available(),
    reason="tiktoken BPE table not reachable in this environment",
)


def test_encoding_name_is_cl100k_base() -> None:
    assert encoding_name() == "cl100k_base"


def test_evidence_budget_is_pinned_at_8000() -> None:
    # Source-pinned. Do not change without a source citation.
    assert EVIDENCE_BUDGET == 8_000


def test_total_context_default_is_not_4096() -> None:
    # 4096 is the R4 threshold and lives ONLY in adapter/faults.py.
    # Ensuring the app's default is different is what proves R4's
    # "learn from the error" requirement is testable at multiple
    # thresholds.
    assert TOTAL_CONTEXT_LIMIT_DEFAULT != 4096
    assert TOTAL_CONTEXT_LIMIT_DEFAULT >= 8000


def test_count_text_returns_zero_for_empty() -> None:
    # Interface-only: no encoding needed for the empty-string fast path.
    assert count_text("") == 0


@needs_tiktoken
def test_count_text_is_nonzero_for_content() -> None:
    assert count_text("hello world") > 0


@needs_tiktoken
def test_count_messages_includes_per_message_overhead() -> None:
    # An empty message is charged the framing tokens + reply primer.
    msgs = [{"role": "user", "content": ""}]
    n = count_messages(msgs)
    # role="user" adds tokens; content="" adds zero. Whatever the
    # exact value, it must be at least TOKENS_PER_MESSAGE +
    # TOKENS_PER_REPLY_PRIMER.
    assert n >= TOKENS_PER_MESSAGE + TOKENS_PER_REPLY_PRIMER


def test_count_messages_empty_list_is_zero() -> None:
    # Interface-only: no encoding needed.
    assert count_messages([]) == 0


def test_count_tool_schemas_none_is_zero() -> None:
    assert count_tool_schemas(None) == 0
    assert count_tool_schemas([]) == 0


@needs_tiktoken
def test_count_tool_schemas_scales_with_content() -> None:
    small = [{"type": "function", "function": {"name": "x"}}]
    big = [
        {
            "type": "function",
            "function": {
                "name": "search_passages",
                "description": "a" * 500,
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        }
    ]
    assert count_tool_schemas(big) > count_tool_schemas(small)


def test_count_output_reservation_positive() -> None:
    assert count_output_reservation(500) == 500


def test_count_output_reservation_none_or_negative_is_zero() -> None:
    assert count_output_reservation(None) == 0
    assert count_output_reservation(-1) == 0
