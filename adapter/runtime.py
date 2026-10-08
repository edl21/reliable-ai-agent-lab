"""Runtime adapter selection for normal and fault-injected runs.

The default path is still the singleton returned by ``adapter.get_model``.
Checks install a ``FaultInjector`` around that same singleton for one
in-process run, then clear it. This keeps fault recipes out of prompts
and prevents routes from constructing parallel model clients.
"""

from __future__ import annotations

from typing import Any

from adapter import get_model
from adapter.clock import Clock, Sleeper
from adapter.faults import FaultConfig, FaultInjector
from adapter.limits import RunLimits

_OVERRIDE: Any = None


def current_model() -> Any:
    """Return the installed run adapter or the normal singleton."""
    return _OVERRIDE if _OVERRIDE is not None else get_model()


def install_fault(
    config: FaultConfig,
    *,
    clock: Clock | None = None,
    sleeper: Sleeper | None = None,
    limits: RunLimits | None = None,
) -> FaultInjector:
    """Install one fresh fault wrapper for a check run."""
    global _OVERRIDE
    injector = FaultInjector(
        get_model(),
        config=config,
        clock=clock,
        sleeper=sleeper,
        limits=limits,
    )
    _OVERRIDE = injector
    return injector


def clear_override() -> None:
    global _OVERRIDE
    _OVERRIDE = None


def current_fault() -> FaultInjector | None:
    return _OVERRIDE if isinstance(_OVERRIDE, FaultInjector) else None


__all__ = ["clear_override", "current_fault", "current_model", "install_fault"]
