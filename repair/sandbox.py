"""Disposable subprocess runner for the immutable repair checks."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VerificationResult:
    exit_code: int
    stdout: str
    stderr: str
    checks_hash_before: str
    checks_hash_after: str
    timed_out: bool = False

    @property
    def success(self) -> bool:
        return (
            self.exit_code == 0
            and any(line == "CHECKS_PASSED" for line in self.stdout.splitlines())
            and self.checks_hash_before == self.checks_hash_after
            and not self.timed_out
        )


class SandboxError(RuntimeError):
    """The runner itself could not enforce its safety invariants."""


class SandboxRunner:
    def __init__(
        self,
        *,
        repair_dir: Path | str = Path(__file__).resolve().parent,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.repair_dir = Path(repair_dir).resolve()
        self.timeout_seconds = timeout_seconds
        self.checks_path = self.repair_dir / "checks.py"
        self.hashes_path = self.repair_dir / "hashes.json"
        self.expected_checks_hash = _load_expected_checks_hash(self.hashes_path)

    def run(self) -> VerificationResult:
        before = sha256_file(self.checks_path)
        if before != self.expected_checks_hash:
            raise SandboxError(
                "immutable checks.py hash does not match repair/hashes.json "
                f"before execution (got {before}, expected "
                f"{self.expected_checks_hash})"
            )
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONIOENCODING": "utf-8",
        }
        for key in list(os.environ):
            if key.startswith("OPENAI_"):
                env.pop(key, None)
        try:
            completed = subprocess.run(
                [sys.executable, "checks.py"],
                cwd=self.repair_dir,
                env=env,
                shell=False,
                timeout=self.timeout_seconds,
                capture_output=True,
                text=True,
            )
            exit_code = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            exit_code = 124
            stdout = _decode(exc.stdout)
            stderr = _decode(exc.stderr) + "\nTIMEOUT"
            timed_out = True
        finally:
            after = sha256_file(self.checks_path)

        if before != after:
            raise SandboxError("immutable checks.py hash changed during execution")
        if after != self.expected_checks_hash:
            raise SandboxError(
                "immutable checks.py hash does not match repair/hashes.json "
                f"after execution (got {after}, expected "
                f"{self.expected_checks_hash})"
            )
        return VerificationResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            checks_hash_before=before,
            checks_hash_after=after,
            timed_out=timed_out,
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_expected_checks_hash(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(payload["checks.py_original_sha256"])


def _decode(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


__all__ = ["SandboxError", "SandboxRunner", "VerificationResult", "sha256_file"]
