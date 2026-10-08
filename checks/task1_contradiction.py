"""Task 1 additional check: contradiction handling.

Adversarial-review-added requirement (Part 1 gap): distinguish a
CHARACTER'S CLAIM from an ESTABLISHED NARRATIVE FACT; surface
unresolved contradictions with citations rather than silently
picking one.

We ask the system to distinguish Ahab's interpretation of Moby Dick
from the narrator's framing rather than flattening a character's belief
into an established fact.

Pipeline-shape assertions only when running against the stub; content
correctness requires a live gateway."""

from __future__ import annotations

import json
import sys

from checks._common import CheckResult, get_client, post_answer


def main() -> int:
    result = CheckResult("task1_contradiction")
    client = get_client()

    status, body, trace_path = post_answer(
        client,
        request_id="task1_contradiction_01",
        question=(
            "Does the narrative establish that Moby Dick is evil, or is "
            "that Ahab's interpretation? Surface conflicting positions."
        ),
        max_chapter=45,
        check_name="contradiction",
    )

    result.check(status == 200, f"HTTP 200 (got {status})")
    result.check(trace_path.exists(), f"trace written to {trace_path}")

    # Schema shape: contradictions[] field must be present in the
    # response (may be empty when running against the stub).
    result.check(
        "contradictions" in body,
        "response schema includes contradictions[] field",
    )
    result.check(
        isinstance(body.get("contradictions"), list),
        f"contradictions is a list (got {type(body.get('contradictions')).__name__})",
    )

    if trace_path.exists():
        events = [json.loads(line) for line in trace_path.read_text().splitlines()]
        result.check(
            any(e["operation"] == "model_call" for e in events),
            "model_call event present (schema instructing contradictions)",
        )

    return result.finish([trace_path])


if __name__ == "__main__":
    sys.exit(main())
