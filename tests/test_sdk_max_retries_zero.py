"""Invariant: the OpenAI SDK client is constructed with ``max_retries=0``.

Rationale (from the adversarial review): the SDK default is
``max_retries=2``. If left at the default, R1 can "succeed" purely
because the SDK retried internally before our own retry loop ran —
breaking R1's virtual-elapsed assertion. Likewise, R2's attempt-count
assertion becomes wrong (the SDK adds hidden attempts).

We enforce this two ways:

1. Directly on the ``ModelAdapter._client`` object we just built,
   which exposes ``max_retries`` through the SDK attribute.
2. As a textual assertion on ``adapter/model.py`` — even if the SDK
   ever hides its ``max_retries`` attribute, the source declaration
   is inspectable.
"""

from __future__ import annotations

import re
from pathlib import Path

from adapter.model import ModelAdapter

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_model_py_declares_max_retries_zero() -> None:
    """Textual guarantee: the constructor kwargs contain
    ``max_retries=0``. This runs even if we're on a Python without
    network access to the gateway."""
    src = (REPO_ROOT / "adapter" / "model.py").read_text(encoding="utf-8")
    # Look for max_retries=0 within a few lines of the OpenAI( call.
    # A very lax pattern is fine — we just want to catch a stray
    # ``max_retries=2`` re-appearing.
    assert re.search(r"max_retries\s*=\s*0", src), (
        "adapter/model.py must construct OpenAI(..., max_retries=0). "
        "The SDK default is 2, which would cause hidden retries and "
        "invalidate R1/R2/R6 assertions."
    )


def test_constructed_client_has_max_retries_zero(monkeypatch) -> None:
    """Actual: construct a ModelAdapter via the private hatch and
    inspect the SDK client. This verifies the SDK honoured our arg."""
    import adapter.model as model_mod

    monkeypatch.setattr(model_mod, "_ALLOW_DIRECT_CONSTRUCTION", True)
    try:
        adapter = ModelAdapter(
            base_url="http://example.invalid/v1",
            api_key="sk-test-fake",
            model_name="stub-model",
        )
    finally:
        monkeypatch.setattr(model_mod, "_ALLOW_DIRECT_CONSTRUCTION", False)

    # The OpenAI SDK stores max_retries on the client under this name.
    # If the SDK renames it in a future release, update BOTH here and
    # in ``adapter/model.py`` — never silently.
    assert adapter.client.max_retries == 0, (
        f"Expected max_retries=0 on the constructed OpenAI client, "
        f"got {adapter.client.max_retries!r}"
    )
