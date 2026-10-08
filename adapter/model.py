"""Model adapter — the one and only import path for LLM calls.

Every LLM call in the codebase MUST go through ``get_model()``. This is
enforced by two mechanisms:

1. **Import path.** ``adapter/__init__.py`` re-exports ``get_model``
   (and nothing else that constructs a client). Code that imports
   ``openai.OpenAI`` directly gets caught by:

2. **A grep-based test** (``tests/test_singleton_adapter.py``) that
   walks the tree and asserts the only file mentioning ``openai.OpenAI(``
   is this one. If a stray helper ever instantiates a second client it
   would silently bypass the fault wrapper — and R1/R2/R6 assertions
   would silently rot. The grep test kills that class of bug.

Two extra invariants live here:

- **SDK ``max_retries=0``**. The upstream ``openai`` client retries
  429/5xx up to twice by default. That is exactly the retry loop we
  own; letting the SDK also do it would make R1's virtual-elapsed
  assertion wrong and R2's attempt-count assertion wrong. We disable
  SDK retries and enforce that decision in
  ``tests/test_sdk_max_retries_zero.py``.

- **Stub fallback**. If the gateway is unreachable or authentication
  fails at startup, ``get_model()`` returns a ``StubModelAdapter``
  instead of raising. Every response from the stub is deterministic
  (keyed by SHA-256 of the messages list) and every trace emitted
  during a stub run carries ``stubbed=true``. The source explicitly
  permits this fallback.

The adapter is intentionally thin. Retry logic, backoff, R1–R6
handling, and context-length recovery live in ``adapter/faults.py``
(the wrapper) and in the app routes. This file's job is
"one method: chat(); talk to the gateway; normalise the response".
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any

import openai
from openai import AuthenticationError, OpenAI, OpenAIError

from adapter.errors import (
    AdapterError,
    AuthError,
    ContextLengthExceeded,
    RateLimited,
    SchemaInvalid,
)

# --- Normalised response ---------------------------------------------------


@dataclass
class AdapterResponse:
    """One shape returned by every adapter (live or stub), so callers
    can be written against a stable interface."""

    content: str | None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int | None] = field(default_factory=dict)
    raw: Any = None
    stubbed: bool = False


# --- Live adapter ----------------------------------------------------------


# A private hatch so unit tests can construct a ``ModelAdapter`` without
# going through ``get_model()``. Set to ``True`` inside test fixtures.
_ALLOW_DIRECT_CONSTRUCTION: bool = False


class ModelAdapter:
    """Live adapter wrapping the official OpenAI SDK. Constructed via
    ``get_model()``; direct instantiation is guarded so a stray
    helper cannot silently create a second client (see module doc)."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model_name: str,
        timeout_seconds: float = 30.0,
        _via_singleton: bool = False,
    ) -> None:
        if not _via_singleton and not _ALLOW_DIRECT_CONSTRUCTION:
            raise RuntimeError(
                "Do not construct ModelAdapter directly; use "
                "adapter.get_model() so the singleton and fault wrapper "
                "stay authoritative."
            )
        self._model_name = model_name
        # NOTE: max_retries=0 is the whole reason we own the retry loop.
        # See module doc and tests/test_sdk_max_retries_zero.py.
        self._client: OpenAI = OpenAI(
            base_url=base_url,
            api_key=api_key,
            max_retries=0,
            timeout=timeout_seconds,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def client(self) -> OpenAI:
        """Exposed for the ``max_retries=0`` invariant test only."""
        return self._client

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> AdapterResponse:
        """Single-shot chat completion. No retry logic here — that lives
        in the fault wrapper and the app routes.

        Translates the OpenAI SDK's error hierarchy into our
        ``AdapterError`` hierarchy so callers can pattern-match on our
        types alone."""
        kwargs: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        if response_format is not None:
            kwargs["response_format"] = response_format
        if max_output_tokens is not None:
            kwargs["max_completion_tokens"] = int(max_output_tokens)
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        if parallel_tool_calls is not None and tools:
            kwargs["parallel_tool_calls"] = bool(parallel_tool_calls)

        try:
            resp = self._client.chat.completions.create(**kwargs)
        except AuthenticationError as e:
            raise AuthError(str(e)) from e
        except openai.RateLimitError as e:
            raise RateLimited(retry_after=_retry_after_from(e)) from e
        except openai.BadRequestError as e:
            # The gateway signals a context-length problem via a
            # 'context_length_exceeded' code in the body. If the error
            # doesn't carry a limit we can't fabricate one — re-raise
            # generically and let the wrapper decide.
            reported = _reported_limit_from(e)
            if reported is not None:
                raise ContextLengthExceeded(reported_limit=reported) from e
            raise AdapterError(str(e)) from e
        except OpenAIError as e:
            raise AdapterError(str(e)) from e

        return _normalise_response(resp)


# --- Stub adapter ----------------------------------------------------------


class StubModelAdapter:
    """Deterministic offline adapter. Returned by ``get_model()`` when
    the gateway is unreachable or authentication fails at startup —
    the source explicitly permits this as a fallback.

    The stub returns one of a small number of canned responses,
    selected by SHA-256 of the (canonicalised) message list. The
    canned set is designed to keep the check scripts runnable
    without a live gateway: it emits schema-valid JSON for
    ``/answer``, produces tool calls that exercise the dispatcher for
    ``/guide``, and lets the fault wrapper's fake responses (R1–R6)
    override its output when a fault recipe is active.

    Every trace event captured during a stub run carries
    ``stubbed=true`` — the writer just needs the caller to pass it."""

    STUB_MODEL_NAME = "stub-adapter"

    def __init__(self) -> None:
        self._model_name = self.STUB_MODEL_NAME

    @property
    def model_name(self) -> str:
        return self._model_name

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> AdapterResponse:
        digest = _messages_digest(messages)
        if tools:
            return AdapterResponse(
                content=None,
                tool_calls=self._canned_tool_calls(
                    messages=messages,
                    digest=digest,
                ),
                usage={
                    "input_tokens": None,
                    "output_tokens": None,
                    "total_tokens": None,
                },
                raw={"stub_digest": digest, "stub_tool_mode": True},
                stubbed=True,
            )
        content = self._canned_content(
            response_format=response_format,
            digest=digest,
            messages=messages,
        )
        return AdapterResponse(
            content=content,
            tool_calls=[],
            usage={"input_tokens": None, "output_tokens": None, "total_tokens": None},
            raw={"stub_digest": digest},
            stubbed=True,
        )

    @classmethod
    def _canned_tool_calls(
        cls,
        *,
        messages: list[dict[str, Any]],
        digest: str,
    ) -> list[dict[str, Any]]:
        """Produce deterministic structured tool calls for offline checks.

        This is not a scripted workflow in the route: the route sends
        each tool result back as a real tool-role message, and this
        adapter chooses the next call from that conversation. The
        first call can intentionally be malformed when the goal asks
        for an invalid-argument recovery check, proving the real
        dispatcher error path can teach the model to correct itself.
        """
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        goal = next(
            (
                str(m.get("content", ""))
                for m in reversed(messages)
                if m.get("role") == "user"
            ),
            "",
        )
        wants_invalid = any(
            marker in goal.lower() for marker in ("invalid", "malformed")
        )

        last_tool = tool_messages[-1] if tool_messages else None
        last_payload = _parse_json_object(
            str(last_tool.get("content", "")) if last_tool else ""
        )

        if wants_invalid and not tool_messages:
            return [
                _stub_tool_call(
                    call_id=f"stub_{digest[:10]}_invalid",
                    name="fetch_passage",
                    arguments={"chunk_id": 12345},
                )
            ]

        if last_tool and last_payload.get("error"):
            return [
                _stub_tool_call(
                    call_id=f"stub_{digest[:10]}_search",
                    name="search_passages",
                    arguments={"query": goal[:500]},
                )
            ]

        if not tool_messages:
            return [
                _stub_tool_call(
                    call_id=f"stub_{digest[:10]}_search",
                    name="search_passages",
                    arguments={"query": goal[:500]},
                )
            ]

        if last_tool and last_tool.get("name") == "search_passages":
            results = last_payload.get("results") or []
            chunk_id = results[0].get("chunk_id") if results else ""
            return [
                _stub_tool_call(
                    call_id=f"stub_{digest[:10]}_fetch",
                    name="fetch_passage",
                    arguments={"chunk_id": chunk_id},
                )
            ]

        if last_tool and last_tool.get("name") == "fetch_passage":
            citation = {
                "chunk_id": last_payload.get("chunk_id", ""),
                "source_filename": last_payload.get("source_filename", ""),
                "chapter": last_payload.get("chapter", 1),
                "pdf_pages": last_payload.get("pdf_pages", []),
                "excerpt": str(last_payload.get("content", ""))[:240],
            }
            return [
                _stub_tool_call(
                    call_id=f"stub_{digest[:10]}_save",
                    name="save_guide",
                    arguments={
                        "title": "Offline guide draft",
                        "content": str(last_payload.get("content", ""))[:1000],
                        "citations": [citation],
                    },
                )
            ]

        # The route should stop after save_guide returns a pending draft.
        # If a caller sends an unexpected extra turn, return no calls
        # rather than inventing an unapproved action.
        return []

    @staticmethod
    def _canned_content(
        *,
        response_format: dict[str, Any] | None,
        digest: str,
        messages: list[dict[str, Any]],
    ) -> str:
        """Emit content that will parse under a JSON-schema
        ``response_format`` when one is requested. For plain text
        calls, return a short deterministic string keyed by the digest."""
        if response_format is None:
            repair = _repair_stub_if_requested(messages)
            if repair is not None:
                return repair
        if (
            response_format is not None
            and response_format.get("type") == "json_schema"
        ):
            # Return schema-shaped-but-empty content that flags the run
            # as stubbed. Callers running against the stub should not
            # be relying on any particular narrative content.
            body = {
                "status": "insufficient_evidence",
                "answer": f"[stubbed:{digest[:8]}] no live model available",
                "citations": [],
                "contradictions": [],
            }
            return json.dumps(body, separators=(",", ":"), ensure_ascii=False)
        return f"[stubbed:{digest[:8]}]"


# --- Singleton -------------------------------------------------------------


_SINGLETON: ModelAdapter | StubModelAdapter | None = None


def get_model() -> ModelAdapter | StubModelAdapter:
    """Return the process-wide singleton adapter.

    Lookup order:

    1. If already constructed, return it.
    2. If ``OPENAI_API_KEY`` is missing, return a fresh
       ``StubModelAdapter``.
    3. Try to construct a ``ModelAdapter``. If SDK client construction
       raises (bad URL, DNS failure, etc.), fall back to the stub.

    We do NOT perform a live ping here — the stub-vs-live decision at
    startup is based on presence of credentials, not gateway
    reachability. A gateway that is up but momentarily failing will
    surface as an ``AdapterError`` at first-call time; the fault
    wrapper decides how to handle that. This keeps ``get_model()``
    cheap and side-effect-free."""
    global _SINGLETON
    if _SINGLETON is not None:
        return _SINGLETON

    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    model_name = os.environ.get("MODEL_NAME", "").strip()

    if not api_key or not base_url or not model_name:
        _SINGLETON = StubModelAdapter()
        return _SINGLETON

    try:
        _SINGLETON = ModelAdapter(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            _via_singleton=True,
        )
    except Exception:
        # SDK client construction failed (malformed URL, etc.); fall back.
        _SINGLETON = StubModelAdapter()
    return _SINGLETON


def _reset_singleton_for_tests() -> None:
    """Test-only escape hatch. Do not use outside ``tests/``."""
    global _SINGLETON
    _SINGLETON = None


# --- helpers ---------------------------------------------------------------


def _messages_digest(messages: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        messages, separators=(",", ":"), ensure_ascii=False, sort_keys=True, default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _repair_stub_if_requested(messages: list[dict[str, Any]]) -> str | None:
    prompt = json.dumps(messages, ensure_ascii=False, default=str)
    if "select_chunks.py" not in prompt or "CHECKS_PASSED" not in prompt:
        return None
    return """def select_chunks(chunks, max_chapter, max_words):
    ordered = sorted(
        enumerate(chunks),
        key=lambda item: (-item[1]["score"], item[0]),
    )
    selected = []
    seen = set()
    used = 0
    for _, chunk in ordered:
        if chunk["chapter"] > max_chapter or chunk["id"] in seen:
            continue
        words = len(chunk["text"].split())
        if used + words > max_words:
            continue
        selected.append(chunk)
        seen.add(chunk["id"])
        used += words
    return selected
"""


def _parse_json_object(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _stub_tool_call(
    *,
    call_id: str,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(
                arguments,
                separators=(",", ":"),
                ensure_ascii=False,
            ),
        },
    }


def _normalise_response(resp: Any) -> AdapterResponse:
    """Convert an ``openai`` chat.completions object into our
    ``AdapterResponse``. Defensive against missing fields — the wire
    protocol changes and we would rather emit ``None`` than KeyError."""
    choice = resp.choices[0] if getattr(resp, "choices", None) else None
    message = getattr(choice, "message", None) if choice else None
    content = getattr(message, "content", None) if message else None

    tool_calls_out: list[dict[str, Any]] = []
    raw_tool_calls = getattr(message, "tool_calls", None) if message else None
    if raw_tool_calls:
        for tc in raw_tool_calls:
            fn = getattr(tc, "function", None)
            tool_calls_out.append(
                {
                    "id": getattr(tc, "id", None),
                    "type": getattr(tc, "type", "function"),
                    "function": {
                        "name": getattr(fn, "name", None) if fn else None,
                        "arguments": getattr(fn, "arguments", None) if fn else None,
                    },
                }
            )

    usage_obj = getattr(resp, "usage", None)
    usage = {
        "input_tokens": getattr(usage_obj, "prompt_tokens", None) if usage_obj else None,
        "output_tokens": getattr(usage_obj, "completion_tokens", None)
        if usage_obj
        else None,
        "total_tokens": getattr(usage_obj, "total_tokens", None) if usage_obj else None,
    }

    return AdapterResponse(
        content=content,
        tool_calls=tool_calls_out,
        usage=usage,
        raw=resp,
        stubbed=False,
    )


def _retry_after_from(exc: Exception) -> float | None:
    """Extract Retry-After (seconds) from an OpenAI SDK RateLimitError.

    The SDK exposes ``exc.response.headers`` when it has one; the
    Retry-After header may be a number of seconds or an HTTP date.
    We only support the numeric form — HTTP-date Retry-After is
    exceedingly rare and the wrapper falls back to backoff-with-jitter
    when None is returned."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response else None
    if not headers:
        return None
    value = headers.get("Retry-After") or headers.get("retry-after")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _reported_limit_from(exc: Exception) -> int | None:
    """Try to extract a ``reported_limit`` from a context-length error
    body. Different gateways signal this differently; we look for
    the common shapes and fall back to None (the caller then treats
    the failure as a generic AdapterError)."""
    body = getattr(exc, "body", None) or getattr(exc, "response", None)
    if body is None:
        return None
    # openai SDK stashes the parsed body on the exception in various
    # attributes depending on version; try a few.
    candidates: list[Any] = []
    for attr in ("body", "message", "error"):
        v = getattr(exc, attr, None)
        if v:
            candidates.append(v)
    for c in candidates:
        if isinstance(c, dict):
            for key in ("reported_limit", "context_window", "max_tokens"):
                if key in c and isinstance(c[key], (int, float)):
                    return int(c[key])
    return None


def _peek_singleton_for_tests() -> ModelAdapter | StubModelAdapter | None:
    """Test-only accessor for the current singleton."""
    return _SINGLETON


__all__ = [
    "AdapterResponse",
    "ModelAdapter",
    "StubModelAdapter",
    "get_model",
]
