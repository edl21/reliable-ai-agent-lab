"""Per-run finite limits, enforced BEFORE every model or tool call.

The source assessment requires that runs stop gracefully with a
structured error when any of the following ceilings is reached — no
fabricated answer, no unbounded retry loops:

- Attempts per operation
- Total model+tool calls in the run
- Elapsed time in the run
- Cumulative model tokens (input + output) in the run

Because ``openai.OpenAI`` is constructed with ``max_retries=0`` (see
``adapter/model.py``), no hidden SDK retry path bypasses this tracker.
Every call goes through ``LimitTracker.pre_call`` first, which either
returns budgets remaining or raises ``LimitExhausted``.

Default values are pinned in the plan and repeated here so the tests
can inspect them without importing README text:

- ``attempts_per_op = 3``
- ``total_calls = 20``
- ``elapsed_seconds = 60`` (virtual in fault tests, real otherwise)
- ``cumulative_tokens = 200_000``

Rationale is documented in ``README.md`` section 9 and in the plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from adapter.clock import Clock, SystemClock
from adapter.errors import LimitExhausted, RetriesExhausted


@dataclass(frozen=True)
class RunLimits:
    """Immutable ceilings for one run. A new run gets a fresh tracker."""

    attempts_per_op: int = 3
    total_calls: int = 20
    elapsed_seconds: float = 60.0
    cumulative_tokens: int = 200_000


@dataclass
class LimitTracker:
    """Enforces a ``RunLimits`` set. Counters reset with each new
    instance — one tracker per run, never reused.

    Distinctions in the errors:

    - ``LimitExhausted`` — the run itself is out of budget (total_calls,
      elapsed_seconds, cumulative_tokens). Terminal for the run.
    - ``RetriesExhausted`` — attempts_per_op was hit on one op; other
      ops in the run could still proceed if the caller chooses.
    """

    limits: RunLimits
    clock: Clock = field(default_factory=SystemClock)

    # counters (mutable)
    calls_used: int = 0
    tokens_used: int = 0
    started_at: float | None = None
    op_attempts: dict[str, int] = field(default_factory=dict)

    def start_run(self) -> None:
        """Latch the run start time. Idempotent — subsequent calls no-op."""
        if self.started_at is None:
            self.started_at = self.clock.now()

    def remaining_deadline(self) -> float:
        """Seconds left before the elapsed budget is exhausted. Zero or
        negative means we're already over budget."""
        if self.started_at is None:
            return self.limits.elapsed_seconds
        elapsed = self.clock.now() - self.started_at
        return self.limits.elapsed_seconds - elapsed

    def pre_call(
        self,
        *,
        op_id: str,
        estimated_input_tokens: int = 0,
        reserved_output_tokens: int = 0,
    ) -> None:
        """Check all budgets before the caller actually invokes the
        model or a tool. Raises the appropriate error type if any
        budget would be exceeded.

        - Bumps ``calls_used`` and ``op_attempts[op_id]`` on success.
        - Reserves the estimated input + reserved output tokens against
          ``tokens_used`` immediately; the caller reconciles with actual
          usage after the call via ``record_actual_tokens``."""
        self.start_run()

        # Attempt cap for this op
        current_attempts = self.op_attempts.get(op_id, 0)
        if current_attempts >= self.limits.attempts_per_op:
            raise RetriesExhausted(
                f"attempts_per_op reached for op={op_id!r} "
                f"({current_attempts}/{self.limits.attempts_per_op})"
            )

        # Total-calls cap
        if self.calls_used >= self.limits.total_calls:
            raise LimitExhausted(
                f"total_calls reached ({self.calls_used}/{self.limits.total_calls})"
            )

        # Elapsed-time cap (must have room to start; the next planned
        # wait is checked separately by the retry policy).
        if self.remaining_deadline() <= 0:
            raise LimitExhausted(
                "elapsed_seconds budget exhausted before this call"
            )

        # Cumulative-tokens cap. We reserve estimated + reserved output
        # up front — the caller reconciles after the call.
        reserved = int(estimated_input_tokens) + int(reserved_output_tokens)
        projected = self.tokens_used + reserved
        if projected > self.limits.cumulative_tokens:
            raise LimitExhausted(
                f"cumulative_tokens would exceed budget: "
                f"{projected} > {self.limits.cumulative_tokens}"
            )

        # All budgets OK — commit.
        self.calls_used += 1
        self.op_attempts[op_id] = current_attempts + 1
        self.tokens_used += reserved

    def record_actual_tokens(
        self, *, reserved: int, actual: int
    ) -> None:
        """Reconcile a token reservation with the model's reported usage."""
        delta = int(actual) - int(reserved)
        self.tokens_used = max(0, self.tokens_used + delta)

    def would_wait_exceed_deadline(self, planned_wait_seconds: float) -> bool:
        """True if adding ``planned_wait_seconds`` would run past the
        deadline. Used by the retry policy to short-circuit rather than
        wait pointlessly (per source: 'If the wait would exceed the
        remaining deadline, stop rather than retry early')."""
        return planned_wait_seconds >= self.remaining_deadline()


__all__ = [
    "LimitTracker",
    "RunLimits",
]
