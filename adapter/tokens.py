"""Token counting with a named tokenizer, and the two independent budgets
the assessment pins.

Design rules (from the plan and the source assessment):

1. **One tokenizer, everywhere.** The app and the R4 fault injector use
   the same functions here so the numbers match. The R4 injector
   rejects requests whose ``total = count_messages(messages) +
   count_tool_schemas(tools) + count_output_reservation(max_output)``
   exceeds its threshold — using anything else would let the app "pass"
   R4 by counting differently from the injector.

2. **Two independent budgets:**

   - ``EVIDENCE_BUDGET = 8_000`` is fixed by the source ("retrieved
     passages + notes + summaries only"). Never negotiated.
   - ``TOTAL_CONTEXT_LIMIT_DEFAULT = 32_000`` is our chosen default and
     is documented in the README. The number 4096 must NOT appear as a
     constant anywhere in this module or the app — the R4 injector owns
     it, and the app learns any reduced limit from the raised
     ``ContextLengthExceeded.reported_limit``.

3. **Documented per-message overhead.** OpenAI-style chat messages
   incur bookkeeping tokens on top of the content itself; we use the
   values documented in the OpenAI Cookbook
   ("How to count tokens with tiktoken", model family: gpt-4/gpt-3.5-turbo):

   - ``TOKENS_PER_MESSAGE = 3`` — the ``<|start|>role\\ncontent<|end|>``
     framing.
   - ``TOKENS_PER_NAME = 1`` — one extra token if the message has a
     ``name`` field.
   - ``TOKENS_PER_REPLY_PRIMER = 3`` — every reply is primed with
     ``<|start|>assistant<|message|>``.

   These values are conservative overestimates for our purposes: if a
   given gateway model uses a slightly different framing, we still won't
   under-count. Documented in README section 7.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# --- Constants pinned by the source or by our locked decisions ---

#: Assessment-pinned: retrieved passages + notes + summaries only. Never grow.
EVIDENCE_BUDGET: int = 8_000

#: Our chosen default. R4 harness overrides this via ContextLengthExceeded.
TOTAL_CONTEXT_LIMIT_DEFAULT: int = 32_000

# --- Per-message overhead (OpenAI Cookbook values for cl100k_base models) ---

TOKENS_PER_MESSAGE: int = 3
TOKENS_PER_NAME: int = 1
TOKENS_PER_REPLY_PRIMER: int = 3

_ENCODING_NAME: str = "cl100k_base"

# Local cache dir for tiktoken's BPE table. Set BEFORE the first
# ``tiktoken.get_encoding`` call so the download (if needed) lands
# under the repo and subsequent runs are offline-safe. The directory
# is created lazily on first use.
_TIKTOKEN_CACHE_DIR = Path(__file__).resolve().parent.parent / ".tiktoken_cache"
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(_TIKTOKEN_CACHE_DIR))

# Encoding is loaded lazily on first use so importing this module does
# NOT force a network fetch. Test environments and offline reviewers
# can still import ``count_text`` etc.; only calling one of the
# counting functions triggers the (one-time) BPE-table load.
_ENC: Any = None


def _get_encoding() -> Any:
    """Return the tiktoken encoding, loading it lazily on first call."""
    global _ENC
    if _ENC is None:
        _TIKTOKEN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        import tiktoken  # imported lazily so the module remains importable offline

        _ENC = tiktoken.get_encoding(_ENCODING_NAME)
    return _ENC


def encoding_name() -> str:
    """Named tokenizer identifier — surfaced in traces and README."""
    return _ENCODING_NAME


def count_text(text: str) -> int:
    """Count tokens in a raw string."""
    if not text:
        return 0
    return len(_get_encoding().encode(text))


def count_messages(messages: list[dict[str, Any]]) -> int:
    """Count tokens in a chat-messages list, including the documented
    per-message overhead and the assistant-reply primer.

    Structure expected (OpenAI chat format):
        [{"role": "system"|"user"|"assistant"|"tool",
          "content": <str | list of content parts>,
          "name": <optional str>,
          "tool_calls": <optional list>,
          "tool_call_id": <optional str>}, ...]

    We stringify structured content parts conservatively: any dict/list
    is JSON-serialised for counting. This overestimates slightly for
    tool-call payloads, which is fine — better than under-counting.
    """
    if not messages:
        return 0

    total = 0
    for msg in messages:
        total += TOKENS_PER_MESSAGE
        for key, value in msg.items():
            if key == "name":
                total += TOKENS_PER_NAME
            if value is None:
                continue
            total += count_text(_as_text(value))
    total += TOKENS_PER_REPLY_PRIMER
    return total


def count_tool_schemas(tools: list[dict[str, Any]] | None) -> int:
    """Count tokens spent on OpenAI-style tool-schema declarations.

    Tools are serialised to JSON on the wire, so a JSON-string encode is
    a fair upper bound. We do not add per-tool framing overhead — the
    schemas themselves already dominate."""
    if not tools:
        return 0
    import json

    return count_text(json.dumps(tools, separators=(",", ":"), ensure_ascii=False))


def count_output_reservation(max_output_tokens: int | None) -> int:
    """Reserved-output tokens counted against the total context.

    We reserve the full ``max_output_tokens`` up front (before we know
    what the model will actually emit). This is what the fault injector
    checks against for R4, so it must match the value the caller passed
    to ``chat(..., max_output_tokens=...)``. ``None`` reserves zero."""
    if max_output_tokens is None or max_output_tokens < 0:
        return 0
    return int(max_output_tokens)


def _as_text(value: Any) -> str:
    """Coerce a message field value into a string for token counting."""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    # Structured content (e.g. tool_calls list, dict payloads) — JSON encode.
    import json

    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, default=str)


__all__ = [
    "EVIDENCE_BUDGET",
    "TOKENS_PER_MESSAGE",
    "TOKENS_PER_NAME",
    "TOKENS_PER_REPLY_PRIMER",
    "TOTAL_CONTEXT_LIMIT_DEFAULT",
    "count_messages",
    "count_output_reservation",
    "count_text",
    "count_tool_schemas",
    "encoding_name",
]
