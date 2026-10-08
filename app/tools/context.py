"""Per-guide tool context and an explicit test-only untrusted-data hook."""

from __future__ import annotations

from dataclasses import dataclass

_TEST_UNTRUSTED_SUFFIX: str | None = None


@dataclass(frozen=True)
class ToolContext:
    """Request-scoped values supplied by the application, never by the
    model. Tools read ``max_chapter`` from here rather than trusting
    tool arguments."""

    request_id: str
    max_chapter: int


def set_test_untrusted_suffix(value: str | None) -> None:
    """Configure a deterministic test-only suffix for fetched passage
    content. This simulates hostile book/tool data; it is never read
    from model output and must be cleared by the check harness."""
    global _TEST_UNTRUSTED_SUFFIX
    _TEST_UNTRUSTED_SUFFIX = value


def get_test_untrusted_suffix() -> str | None:
    return _TEST_UNTRUSTED_SUFFIX


__all__ = [
    "ToolContext",
    "get_test_untrusted_suffix",
    "set_test_untrusted_suffix",
]
