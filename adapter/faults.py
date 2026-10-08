"""Fault-injection wrapper — SKELETON for Phase 1.

Full R1–R6 recipes land in Phase 4. This file exists in Phase 1 so
that every LLM caller written between now and then can already thread
its calls through the wrapper: when the wrapper is constructed with an
empty ``FaultConfig`` it is a strict pass-through, and the live/stub
adapter path runs unmodified. That is what proves the source's
"wrapper, not replacement" requirement.

Design invariants set here (and honoured by Phase 4):

- **Failure recipes are selected via ``FaultConfig``** supplied by the
  check harness — never derived from model output or from the request
  body. The wrapper cannot be turned on by a prompt-injected string.
- **Time is injectable.** Real code uses ``SystemClock`` /
  ``SystemSleeper``. Fault tests use ``VirtualClock`` /
  ``VirtualSleeper`` — no real ``time.sleep`` in any fault test path.
- **The wrapper wraps** the ``ModelAdapter`` (and, in Phase 3+, the
  tool dispatcher). It never re-implements the SDK call. That keeps
  R1–R6 focused on the code path the app actually runs.

The Phase 4 additions will slot into the ``chat`` method here; the
call surface stays identical so Phases 2 and 3 can be written
against the wrapper today without knowing the eventual recipes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from adapter.clock import Clock, Sleeper, SystemClock, SystemSleeper
from adapter.errors import (
    AuthError,
    ContextLengthExceeded,
    RateLimited,
)
from adapter.limits import LimitTracker, RunLimits
from adapter.tokens import (
    count_messages,
    count_output_reservation,
    count_tool_schemas,
)

if TYPE_CHECKING:  # pragma: no cover — import for type-check only
    from adapter.model import ModelAdapter, StubModelAdapter
from adapter.model import AdapterResponse


# --- FaultConfig -----------------------------------------------------------


@dataclass
class FaultConfig:
    """Which fault recipe (if any) is active for this run.

    All fields default to "no injection". An empty ``FaultConfig()`` is
    the production shape and turns the wrapper into a strict
    pass-through.

    The recipe fields will be consumed by ``FaultInjector.chat`` in
    Phase 4. They are declared here so that:

    - the plumbing that constructs the wrapper is complete today, and
    - Phase-4 wiring is a mechanical fill-in of the recipe bodies,
      not a re-plumbing of every call site.
    """

    #: R1: raise 429+Retry-After on first call, then succeed.
    r1_first_call_429: bool = False
    r1_retry_after_seconds: float | None = 2.0

    #: R2: raise 429+Retry-After on every call.
    r2_all_calls_429: bool = False
    r2_retry_after_seconds: float | None = 2.0

    #: R3: on the target save operation, commit then raise TimeoutError.
    r3_save_then_timeout: bool = False
    r3_target_operation_id: str | None = None

    #: R4: raise ContextLengthExceeded when total tokens exceed the
    #: threshold. Only threshold lives here — the app must NEVER
    #: read it from anywhere else.
    r4_context_threshold: int | None = None

    #: R5: first call returns literal {"status": 42}; second call
    #: passes through.
    r5_first_call_bad_schema: bool = False

    #: R6: raise AuthError on every call.
    r6_all_calls_401: bool = False

    #: Deterministic seed used by the shared recovery helper when a
    #: recipe omits Retry-After and jitter is required.
    jitter_seed: int = 0

    #: Internal counter for recipes that trigger on the first call only.
    #: The check harness resets this by constructing a fresh config
    #: for each run (source: "reset counters between runs").
    _call_count: int = field(default=0, repr=False)
    _r3_timeout_raised: bool = field(default=False, repr=False)

    def is_empty(self) -> bool:
        """True when no recipe is active (production/live path shape)."""
        return not (
            self.r1_first_call_429
            or self.r2_all_calls_429
            or self.r3_save_then_timeout
            or self.r4_context_threshold is not None
            or self.r5_first_call_bad_schema
            or self.r6_all_calls_401
        )


# --- FaultInjector ---------------------------------------------------------


class FaultInjector:
    """Wraps a live or stub adapter and (in Phase 4) intercepts the
    real call sites to substitute injected failures.

    Constructed once per run by the harness. Empty ``FaultConfig`` =
    pass-through, which is the shape used in production.

    Phase 1 responsibilities (implemented here):

    - Present the same ``.chat(...)`` signature as ``ModelAdapter``.
    - Delegate straight through to the underlying adapter.
    - Own the clock and sleeper — even in pass-through mode — so
      Phase 4 can start rejecting/retrying without re-plumbing.

    Phase 4 responsibilities (stubs raise NotImplementedError inside
    the private helpers below so a future call finds them cleanly)."""

    def __init__(
        self,
        inner: "ModelAdapter | StubModelAdapter",
        *,
        config: FaultConfig | None = None,
        clock: Clock | None = None,
        sleeper: Sleeper | None = None,
        limits: RunLimits | None = None,
    ) -> None:
        self._inner = inner
        self._config = config or FaultConfig()
        self._clock: Clock = clock or SystemClock()
        self._sleeper: Sleeper = sleeper or SystemSleeper()
        self._tracker = LimitTracker(
            limits=limits or RunLimits(),
            clock=self._clock,
        )

    # --- accessors used by tests / Phase 4 ---

    @property
    def config(self) -> FaultConfig:
        return self._config

    @property
    def clock(self) -> Clock:
        return self._clock

    @property
    def sleeper(self) -> Sleeper:
        return self._sleeper

    @property
    def inner(self) -> "ModelAdapter | StubModelAdapter":
        """The wrapped adapter. Exposed so callers can assert the
        live path is still reachable (source: 'Keep the real/live
        adapter path fully intact and reachable — this is a wrapper,
        not a replacement.')."""
        return self._inner

    @property
    def tracker(self) -> LimitTracker:
        return self._tracker

    @property
    def is_stubbed(self) -> bool:
        return bool(getattr(self._inner, "STUB_MODEL_NAME", False))

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    # --- call surface (Phase 1 = pass-through) ---

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
        """Intercept configured failures, otherwise call the same inner
        adapter used by live and stub production paths."""
        self._config._call_count += 1
        call_no = self._config._call_count

        if self._config.r6_all_calls_401:
            raise AuthError("[injected R6] HTTP 401")
        if self._config.r2_all_calls_429:
            raise RateLimited(
                retry_after=self._config.r2_retry_after_seconds,
                message="[injected R2] HTTP 429",
            )
        if self._config.r1_first_call_429 and call_no == 1:
            raise RateLimited(
                retry_after=self._config.r1_retry_after_seconds,
                message="[injected R1] first-call HTTP 429",
            )

        if (
            self._config.r4_context_threshold is not None
            and (
                count_messages(messages)
                + count_tool_schemas(tools)
                + count_output_reservation(max_output_tokens)
                > self._config.r4_context_threshold
            )
        ):
            raise ContextLengthExceeded(
                reported_limit=self._config.r4_context_threshold,
                message=(
                    "[injected R4] context_length_exceeded; "
                    f"reported_limit={self._config.r4_context_threshold}"
                ),
            )

        if (
            self._config.r5_first_call_bad_schema
            and call_no == 1
            and response_format is not None
        ):
            return AdapterResponse(
                content='{"status": 42}',
                raw={"injected": "R5", "response": '{"status": 42}'},
                stubbed=self.is_stubbed,
            )

        return self._inner.chat(
            messages=messages,
            tools=tools,
            response_format=response_format,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
        )

    def after_approval_write(self, operation_id: str) -> None:
        """R3 commits through the real ledger first, then makes the
        caller observe a timeout. A replay sees the saved ledger row."""
        if (
            self._config.r3_save_then_timeout
            and self._config.r3_target_operation_id == operation_id
            and not self._config._r3_timeout_raised
        ):
            self._config._r3_timeout_raised = True
            raise TimeoutError("[injected R3] response timed out after commit")


__all__ = [
    "FaultConfig",
    "FaultInjector",
]
