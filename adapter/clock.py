"""Injectable clock and sleeper.

Task 3 (R1, R2) requires that we assert on virtual elapsed time without
ever calling ``time.sleep`` inside a test. The rule pinned in the plan
is: **no fault test path ever calls the real sleeper**. Real code paths
receive a real ``SystemClock``/``SystemSleeper``; fault tests receive a
``VirtualClock``/``VirtualSleeper``, and the trace event records
``virtual=True`` with the advanced ``virtual_elapsed_ms``.

The ``Clock`` and ``Sleeper`` types are ``Protocol``s so we can supply
plain classes at runtime and avoid inheritance chains.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol


class Clock(Protocol):
    """A monotonic time source. Return seconds since some epoch — only
    differences matter, never the absolute value."""

    def now(self) -> float: ...


class Sleeper(Protocol):
    """A sleep primitive. ``sleep(seconds)`` MUST advance whatever clock
    the sleeper shares state with. In production this maps to
    ``time.sleep``; in tests it advances a shared virtual clock."""

    def sleep(self, seconds: float) -> None: ...


class SystemClock:
    """Real monotonic clock. Used by the live adapter path in production."""

    def now(self) -> float:
        return time.monotonic()


class SystemSleeper:
    """Real sleeper. Never used in fault tests."""

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


@dataclass
class VirtualClock:
    """A clock whose ``now()`` reads a counter advanced only by an
    associated ``VirtualSleeper``. Zero real time passes.

    Rationale: R1 must show virtual_elapsed >= 2s within a test that
    completes in milliseconds. R2 must show that the run stops when the
    next planned wait would exceed the remaining deadline — testing
    that with real ``time.sleep`` would either take minutes or be a
    lie about what the code did.
    """

    _now: float = 0.0

    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("cannot advance clock backwards")
        self._now += seconds


@dataclass
class VirtualSleeper:
    """A sleeper that advances a paired ``VirtualClock`` and records the
    total requested-and-served sleep duration for trace assertions."""

    clock: VirtualClock
    total_slept: float = 0.0
    history: list[float] = field(default_factory=list)

    def sleep(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("cannot sleep negative seconds")
        self.history.append(seconds)
        self.total_slept += seconds
        self.clock.advance(seconds)


__all__ = [
    "Clock",
    "Sleeper",
    "SystemClock",
    "SystemSleeper",
    "VirtualClock",
    "VirtualSleeper",
]
