"""Sanctioned import surface for the adapter package.

Callers should import ``get_model`` from here. Do not import
``ModelAdapter`` and do not construct an SDK client anywhere else in
the codebase.

The grep-based invariant test at ``tests/test_singleton_adapter.py``
enforces the constraint: it asserts that the only file constructing
an SDK client is ``adapter/model.py``. That is what makes the fault
wrapper authoritative: every LLM call goes through ``get_model()``,
and every run in a fault-injected harness threads through the
``FaultInjector`` that wraps the returned adapter.
"""

from adapter.model import get_model

__all__ = ["get_model"]
