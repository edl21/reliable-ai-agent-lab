"""Task 1 required check: two-chapter question with FULL ordered
model messages in the trace.

Source (verbatim): *"Two-chapter question (include full ordered model
messages, credentials omitted)."* — we exercise that by asking a
question that plausibly spans two chapters and asserting the trace
records the ordered ``[system, user]`` message shapes."""

from __future__ import annotations

import json
import sys

from checks._common import CheckResult, get_client, post_answer


def main() -> int:
    result = CheckResult("task1_two_chapter")
    client = get_client()

    status, body, trace_path = post_answer(
        client,
        request_id="task1_two_01",
        question=(
            "Compare Ishmael's reason for going to sea with his first "
            "impressions of Queequeg."
        ),
        max_chapter=4,
        check_name="two_chapter",
    )

    result.check(status == 200, f"HTTP 200 (got {status})")
    result.check(trace_path.exists(), f"trace written to {trace_path}")

    if trace_path.exists():
        events = [json.loads(line) for line in trace_path.read_text().splitlines()]
        model_events = [e for e in events if e["operation"] == "model_call"]
        result.check(
            len(model_events) >= 1, "at least one model_call event"
        )
        if model_events:
            notes = model_events[0].get("notes", {})
            msgs = notes.get("ordered_messages", [])
            result.check(
                len(msgs) >= 2,
                f"ordered messages present (n={len(msgs)})",
            )
            if msgs:
                result.check(
                    msgs[0].get("role") == "system",
                    f"first message is system (got {msgs[0].get('role')!r})",
                )
                result.check(
                    msgs[-1].get("role") == "user",
                    f"last message is user (got {msgs[-1].get('role')!r})",
                )
                # Credential-omission smoke: full message content must
                # not contain "sk-" or "Bearer " tokens.
                for m in msgs:
                    content = str(m.get("content", ""))
                    result.check(
                        "sk-" not in content and "Bearer " not in content,
                        f"no credential-like tokens in message content ({m.get('role')})",
                    )

    return result.finish([trace_path])


if __name__ == "__main__":
    sys.exit(main())
