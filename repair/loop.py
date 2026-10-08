"""Bounded model-feedback repair state machine for Task 4."""

from __future__ import annotations

import difflib
import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adapter.runtime import current_model
from repair.allowlist import AllowlistResult, validate_source
from repair.sandbox import SandboxError, SandboxRunner, VerificationResult, sha256_file
from traces.schema import JSONLWriter, TraceEvent


BEHAVIOR_SPEC = (
    "Implement the general rule: inclusive chapter limit (keep chunks with "
    "chapter <= max_chapter); inclusive word budget; descending score order "
    "with original-order tie-breaking (use enumerate + sorted, not "
    "list.sort — mutating the input list fails checks); skip a too-large "
    "chunk with continue (not break) so later smaller chunks can still fit; "
    "skip duplicate IDs after first selection; never mutate input list or "
    "dictionaries; whitespace-split word counting. Do not hardcode the "
    "fixtures or fabricate CHECKS_PASSED."
)

#: Explicit allowlist constraints — live models often invent .get()/max()/copy().
ALLOWLIST_CONSTRAINTS = (
    "Allowlist (hard reject before checks run):\n"
    "- Allowed calls/methods only: sorted, len, set, enumerate, sum, range, "
    "list, tuple, split, append, add.\n"
    "- Access dict fields with brackets only: chunk['id'], chunk['chapter'], "
    "chunk['score'], chunk['text']. Never use .get, .keys, .values, .items.\n"
    "- Never use list.sort / chunks.sort (mutates input). Use "
    "sorted(enumerate(chunks), key=lambda item: (-item[1]['score'], item[0])).\n"
    "- Never use max, min, copy, deepcopy, dict(...), open, eval, exec, "
    "imports, or attribute access other than split/append/add.\n"
    "- Exactly one top-level function named select_chunks; no imports."
)

REPAIR_SYSTEM_PROMPT = (
    "You repair one Python utility. Return ONLY the complete replacement "
    "source for select_chunks.py inside a python code fence.\n"
    "Critical constraints:\n"
    "1) Use sorted(enumerate(chunks), ...) for score order + stable ties; "
    "never chunks.sort or list.sort.\n"
    "2) Use continue (not break) when a chunk exceeds the remaining word budget.\n"
    "3) Inclusive chapter test: keep when chunk['chapter'] <= max_chapter.\n"
    "4) Dict access via brackets only — never .get().\n"
    "5) No imports, I/O, processes, dynamic execution, max/min/copy, or "
    "other disallowed calls."
)

SUPPLIED_BROKEN_SOURCE = """\
def select_chunks(chunks, max_chapter, max_words):
    chunks.sort(key=lambda c: c["score"], reverse=True)
    selected = []
    seen = set()
    used = 0
    for chunk in chunks:
        if chunk["chapter"] >= max_chapter or chunk["id"] in seen:
            continue
        words = len(chunk["text"].split())
        if used + words > max_words:
            break
        selected.append(chunk)
        seen.add(chunk["id"])
        used += words
    return selected
"""


class BaselineMismatchError(RuntimeError):
    """The utility under repair is not the approved broken baseline."""


@dataclass
class RepairResult:
    success: bool
    attempts: int
    trace_path: Path
    final_error: str = ""


class RepairRunner:
    def __init__(
        self,
        *,
        repair_dir: Path | str = Path(__file__).resolve().parent,
        max_edits: int = 3,
        model: Any | None = None,
        reset_to_supplied: bool = False,
    ) -> None:
        self.repair_dir = Path(repair_dir).resolve()
        self.utility_path = self.repair_dir / "select_chunks.py"
        self.checks_path = self.repair_dir / "checks.py"
        self.hashes_path = self.repair_dir / "hashes.json"
        self.max_edits = max_edits
        self.model = model or current_model()
        self.sandbox = SandboxRunner(repair_dir=self.repair_dir)
        self.reset_to_supplied = reset_to_supplied
        self._hashes = json.loads(self.hashes_path.read_text(encoding="utf-8"))

    def run(self) -> RepairResult:
        run_id = uuid.uuid4().hex
        trace_path = self.repair_dir / "traces" / f"repair_{run_id}.jsonl"
        writer = JSONLWriter()
        snapshot_dir = self.repair_dir / "snapshots" / run_id
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        if self.reset_to_supplied:
            self.utility_path.write_text(
                SUPPLIED_BROKEN_SOURCE,
                encoding="utf-8",
            )

        expected_utility = str(self._hashes["select_chunks.py_original_sha256"])
        expected_checks = str(self._hashes["checks.py_original_sha256"])
        utility_hash = sha256_file(self.utility_path)
        checks_hash = sha256_file(self.checks_path)
        if utility_hash != expected_utility:
            raise BaselineMismatchError(
                "repair requires the approved broken utility baseline from "
                f"hashes.json (got {utility_hash}, expected {expected_utility}). "
                "Pass reset_to_supplied=True to restore the demonstration baseline."
            )
        if checks_hash != expected_checks:
            raise SandboxError(
                "immutable checks.py hash does not match repair/hashes.json "
                f"(got {checks_hash}, expected {expected_checks})"
            )

        original_checks_hash = checks_hash
        current_source = self.utility_path.read_text(encoding="utf-8")

        # S0: initial run does not count as an edit.
        initial = self.sandbox.run()
        self._record(
            writer,
            trace_path,
            run_id,
            attempt=0,
            outcome="ok" if initial.success else "error",
            notes={
                "state": "S0_initial_run",
                "exit_code": initial.exit_code,
                "stdout": initial.stdout,
                "stderr": initial.stderr,
                "checks_hash_before": initial.checks_hash_before,
                "checks_hash_after": initial.checks_hash_after,
                "utility_baseline_hash": utility_hash,
                "expected_utility_baseline_hash": expected_utility,
                "initial_failure_does_not_count": True,
                "reset_to_supplied": self.reset_to_supplied,
            },
        )
        if initial.success:
            return RepairResult(True, 0, trace_path)

        last_sandbox_error = _failure_text(initial)
        last_error = last_sandbox_error
        attempts = 0
        while attempts < self.max_edits:
            attempt = attempts + 1
            prompt = self._prompt(current_source, last_error)
            try:
                response = self.model.chat(
                    messages=[
                        {
                            "role": "system",
                            "content": REPAIR_SYSTEM_PROMPT,
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_output_tokens=3_000,
                    temperature=0.0,
                )
            except Exception as exc:
                attempts += 1
                last_error = (
                    f"model_call_failed: {exc!r}\n\n"
                    f"Previous sandbox failure still applies:\n"
                    f"{last_sandbox_error}"
                )
                self._record(
                    writer,
                    trace_path,
                    run_id,
                    attempt=attempt,
                    outcome="error",
                    notes={
                        "state": "S1_model_error",
                        "error": repr(exc),
                        "last_sandbox_error": last_sandbox_error,
                    },
                )
                continue
            attempts += 1
            candidate = _extract_source(response.content or "")
            validation = validate_source(candidate)
            snapshot = snapshot_dir / f"attempt_{attempt}.py"
            snapshot.write_text(candidate, encoding="utf-8")
            diff = _diff(current_source, candidate)
            (snapshot_dir / f"attempt_{attempt}.diff").write_text(
                diff,
                encoding="utf-8",
            )
            self._record(
                writer,
                trace_path,
                run_id,
                attempt=attempt,
                outcome="ok" if validation.allowed else "rejected",
                notes={
                    "state": "S2_validate_patch",
                    "model_request": prompt,
                    "model_response": response.content,
                    "stubbed": bool(getattr(response, "stubbed", False)),
                    "allowlist_allowed": validation.allowed,
                    "allowlist_reason": validation.reason,
                    "diff": diff,
                    "snapshot": str(snapshot),
                },
            )
            if not validation.allowed:
                # Keep sandbox FAIL detail so the next attempt still sees
                # which behavioural cases remain broken, plus a concrete
                # rewrite hint for common allowlist mistakes.
                last_error = (
                    f"allowlist_rejected: {validation.reason}\n"
                    f"{_allowlist_rewrite_hint(validation.reason)}\n\n"
                    f"Previous sandbox failure still applies:\n"
                    f"{last_sandbox_error}"
                )
                continue

            # S3: apply only the utility; checks.py remains untouched.
            self.utility_path.write_text(candidate, encoding="utf-8")
            try:
                verification = self.sandbox.run()
            except Exception as exc:
                last_error = repr(exc)
                last_sandbox_error = last_error
                self._record(
                    writer,
                    trace_path,
                    run_id,
                    attempt=attempt,
                    outcome="error",
                    notes={
                        "state": "S4_sandbox_error",
                        "error": last_error,
                        "checks_hash_original": original_checks_hash,
                        "checks_hash_final": sha256_file(self.checks_path),
                    },
                )
                continue

            self._record(
                writer,
                trace_path,
                run_id,
                attempt=attempt,
                outcome="ok" if verification.success else "error",
                notes={
                    "state": "S5_evaluate_result",
                    "exit_code": verification.exit_code,
                    "stdout": verification.stdout,
                    "stderr": verification.stderr,
                    "checks_hash_before": verification.checks_hash_before,
                    "checks_hash_after": verification.checks_hash_after,
                    "checks_hash_original": original_checks_hash,
                    "checks_hash_unchanged": (
                        verification.checks_hash_before
                        == original_checks_hash
                        == verification.checks_hash_after
                    ),
                    "diff": diff,
                },
            )
            if verification.success:
                self._record(
                    writer,
                    trace_path,
                    run_id,
                    attempt=attempt,
                    outcome="ok",
                    notes={
                        "state": "Sy_success",
                        "final_utility": self.utility_path.read_text(
                            encoding="utf-8"
                        ),
                        "checks_hash_original": original_checks_hash,
                        "checks_hash_final": sha256_file(self.checks_path),
                    },
                )
                return RepairResult(True, attempts, trace_path)

            current_source = candidate
            last_sandbox_error = _failure_text(verification)
            last_error = last_sandbox_error

        self._record(
            writer,
            trace_path,
            run_id,
            attempt=attempts,
            outcome="error",
            recovery_decision="limit_exhausted",
            notes={
                "state": "Sx_attempt_exhausted",
                "attempts": attempts,
                "max_edits": self.max_edits,
                "last_error": last_error,
                "final_utility": self.utility_path.read_text(encoding="utf-8"),
                "checks_hash_original": original_checks_hash,
                "checks_hash_final": sha256_file(self.checks_path),
            },
        )
        return RepairResult(False, attempts, trace_path, last_error)

    @staticmethod
    def _prompt(current_source: str, failure: str) -> str:
        return (
            "Current select_chunks.py:\n"
            "```python\n"
            f"{current_source}\n"
            "```\n\n"
            f"Behavioral specification:\n{BEHAVIOR_SPEC}\n\n"
            f"{ALLOWLIST_CONSTRAINTS}\n\n"
            "Initial or latest checks.py output / validation feedback:\n"
            f"{failure}\n\n"
            "Return only a complete replacement select_chunks.py that "
            "satisfies both the behavioural rules and the allowlist."
        )

    @staticmethod
    def _record(
        writer: JSONLWriter,
        path: Path,
        run_id: str,
        *,
        attempt: int,
        outcome: str,
        notes: dict[str, Any],
        recovery_decision: str = "none",
    ) -> None:
        writer.append(
            path,
            TraceEvent(
                request_id=run_id,
                run_id=run_id,
                task="t4",
                operation="repair_step",
                attempt=attempt,
                outcome=outcome,  # type: ignore[arg-type]
                recovery_decision=recovery_decision,  # type: ignore[arg-type]
                stubbed=bool(notes.get("stubbed", False)),
                notes=notes,
            ),
        )


def _allowlist_rewrite_hint(reason: str) -> str:
    """Map common allowlist rejects to a concrete next-edit instruction."""
    lowered = (reason or "").lower()
    if "disallowed call: get" in lowered or "disallowed attribute: get" in lowered:
        return (
            "Hint: replace every .get(...) with bracket access, e.g. "
            "chunk['score'] / chunk['chapter'] / chunk['id'] / chunk['text']."
        )
    if "disallowed call: sort" in lowered or "mutat" in lowered:
        return (
            "Hint: do not call .sort on chunks. Build a new order with "
            "sorted(enumerate(chunks), key=lambda item: "
            "(-item[1]['score'], item[0]))."
        )
    if "disallowed call: max" in lowered or "disallowed call: min" in lowered:
        return (
            "Hint: max/min are not allowlisted. Express comparisons with "
            "ordinary if / >/</<= operators instead."
        )
    if "disallowed call: copy" in lowered or "deepcopy" in lowered:
        return (
            "Hint: copy/deepcopy are not allowlisted. Build a new selected "
            "list with append; never mutate the input chunks list."
        )
    if "disallowed call:" in lowered or "disallowed attribute:" in lowered:
        return (
            "Hint: stick to sorted/len/set/enumerate/sum/range/list/tuple "
            "and methods split/append/add only; use chunk['field'] brackets."
        )
    return "Hint: rewrite using only allowlisted calls and bracket dict access."


def _extract_source(content: str) -> str:
    """Accept a complete source response, optionally inside a code fence."""
    match = re.search(r"```(?:python)?\s*(.*?)```", content, flags=re.DOTALL)
    if match:
        return match.group(1).strip() + "\n"
    return content.strip() + "\n"


def _diff(before: str, after: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile="select_chunks.py.before",
            tofile="select_chunks.py.after",
        )
    )


def _failure_text(result: VerificationResult) -> str:
    return (
        f"exit_code={result.exit_code}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}\n"
        f"checks_hash_before={result.checks_hash_before}\n"
        f"checks_hash_after={result.checks_hash_after}"
    )


__all__ = [
    "ALLOWLIST_CONSTRAINTS",
    "BEHAVIOR_SPEC",
    "BaselineMismatchError",
    "REPAIR_SYSTEM_PROMPT",
    "RepairResult",
    "RepairRunner",
    "SUPPLIED_BROKEN_SOURCE",
]
