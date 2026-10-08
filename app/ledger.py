"""In-memory approval/write ledger for Task 2 and R3.

The ledger is the idempotency boundary:

pending draft -> approve -> exactly one application-generated artifact
-> replay returns the same artifact ID without another write.

The canonical payload hash is produced only by ``app.canonical`` and
contains ``guide + citations + chapter_limit``. ``payload_check`` on
``/approve`` is optional for the normal source-shaped request, but
lets tests prove that reusing an operation ID with changed content is
rejected.

Artifact writes go exclusively through ``_commit``, which requires and
validates ``operation_id``, canonical ``payload_hash``, and the
server-minted ``approval_token`` immediately before writing.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.canonical import canonical_hash


class LedgerError(RuntimeError):
    """Base for approval-ledger failures."""


class UnknownOperation(LedgerError):
    """The operation ID does not exist."""


class OperationPayloadTampered(LedgerError):
    """The replay supplied a different payload hash."""


class OperationNotApprovable(LedgerError):
    """The operation is already rejected or otherwise invalid."""


class ApprovalTokenInvalid(LedgerError):
    """A direct commit attempt presented a missing or mismatched token."""


@dataclass
class WriteRecord:
    operation_id: str
    artifact_id: str
    path: str


@dataclass
class ApprovalRecord:
    operation_id: str
    payload_hash: str
    draft: dict[str, Any]
    state: str = "pending"
    artifact_id: str | None = None
    approval_token: str | None = field(default=None, repr=False)
    created_at: str = ""
    rejected_at: str | None = None


class WriteLog:
    """Append-only in-memory write log used to assert exactly-one-write."""

    def __init__(self) -> None:
        self._records: list[WriteRecord] = []

    def append(self, record: WriteRecord) -> None:
        self._records.append(record)

    def for_operation(self, operation_id: str) -> list[WriteRecord]:
        return [r for r in self._records if r.operation_id == operation_id]

    def count(self, operation_id: str) -> int:
        return len(self.for_operation(operation_id))

    def all(self) -> list[WriteRecord]:
        return list(self._records)

    def clear(self) -> None:
        self._records.clear()


class ApprovalLedger:
    """Thread-safe in-memory ledger with app-generated artifact files."""

    def __init__(
        self,
        *,
        artifact_root: Path | str = Path("artifacts"),
        server_secret: str | None = None,
    ) -> None:
        self.artifact_root = Path(artifact_root)
        self._server_secret = server_secret or os.urandom(32).hex()
        self._records: dict[str, ApprovalRecord] = {}
        self.write_log = WriteLog()
        self._lock = threading.RLock()

    def create_pending(
        self,
        *,
        title: str,
        content: str,
        citations: list[dict[str, Any]],
        chapter_limit: int,
    ) -> ApprovalRecord:
        """Create a pending operation; no artifact is written."""
        draft = {
            "title": title,
            "content": content,
            "citations": citations,
            "chapter_limit": chapter_limit,
        }
        operation_id = f"op_{uuid.uuid4().hex}"
        record = ApprovalRecord(
            operation_id=operation_id,
            payload_hash=canonical_hash(draft),
            draft=draft,
            created_at=str(uuid.uuid4()),
        )
        with self._lock:
            self._records[operation_id] = record
        return record

    def get(self, operation_id: str) -> ApprovalRecord:
        with self._lock:
            try:
                return self._records[operation_id]
            except KeyError as exc:
                raise UnknownOperation(operation_id) from exc

    def approve(
        self,
        *,
        operation_id: str,
        approve: bool,
        payload_check: str | None = None,
    ) -> tuple[ApprovalRecord, bool]:
        """Approve/reject/replay an operation.

        Returns ``(record, replayed)``. A replayed saved operation never
        writes again. The lock covers the state check and write-log
        append, so concurrent duplicate approvals cannot double-write.
        """
        with self._lock:
            record = self.get(operation_id)
            if payload_check is not None and payload_check != record.payload_hash:
                raise OperationPayloadTampered(operation_id)

            if record.state == "saved":
                return record, True
            if record.state == "rejected":
                raise OperationNotApprovable(
                    f"operation {operation_id} was rejected"
                )
            if not approve:
                record.state = "rejected"
                record.rejected_at = str(uuid.uuid4())
                return record, False

            # Mint the approval token, then commit only through the
            # validated internal write path.
            approval_token = self._make_approval_token(record)
            record.approval_token = approval_token
            self._commit(
                operation_id=record.operation_id,
                payload_hash=record.payload_hash,
                approval_token=approval_token,
            )
            return record, False

    def _commit(
        self,
        *,
        operation_id: str,
        payload_hash: str,
        approval_token: str | None,
    ) -> ApprovalRecord:
        """Sole artifact write path.

        Requires a matching operation ID, canonical payload hash, and
        server-minted approval token. Missing or mismatched tokens fail
        closed with no write. Safe to call while ``approve`` already
        holds the reentrant lock.
        """
        if not operation_id:
            raise ApprovalTokenInvalid("missing operation_id")
        if not payload_hash:
            raise ApprovalTokenInvalid("missing payload_hash")
        if not approval_token:
            raise ApprovalTokenInvalid("missing approval_token")

        with self._lock:
            record = self.get(operation_id)
            if payload_hash != record.payload_hash:
                raise OperationPayloadTampered(operation_id)
            expected = self._make_approval_token(record)
            if approval_token != expected:
                raise ApprovalTokenInvalid("approval_token mismatch")
            if (
                record.approval_token is not None
                and approval_token != record.approval_token
            ):
                raise ApprovalTokenInvalid("approval_token mismatch")
            if record.state == "saved":
                return record
            if record.state == "rejected":
                raise OperationNotApprovable(
                    f"operation {operation_id} was rejected"
                )

            artifact_id = f"guide_{uuid.uuid4().hex}"
            path = self.artifact_root / f"{artifact_id}.json"
            self.artifact_root.mkdir(parents=True, exist_ok=True)
            artifact = {
                "artifact_id": artifact_id,
                "operation_id": record.operation_id,
                "payload_hash": record.payload_hash,
                **record.draft,
            }
            path.write_text(
                json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            self.write_log.append(
                WriteRecord(
                    operation_id=record.operation_id,
                    artifact_id=artifact_id,
                    path=str(path),
                )
            )
            record.approval_token = approval_token
            record.artifact_id = artifact_id
            record.state = "saved"
            return record

    def reset(self) -> None:
        """Test-only reset; removes ledger state and generated artifacts."""
        with self._lock:
            for record in self.write_log.all():
                Path(record.path).unlink(missing_ok=True)
            if self.artifact_root.exists():
                for path in self.artifact_root.glob("*.json"):
                    path.unlink(missing_ok=True)
            self._records.clear()
            self.write_log.clear()

    def _make_approval_token(self, record: ApprovalRecord) -> str:
        material = (
            f"{record.operation_id}:{record.payload_hash}:{self._server_secret}"
        ).encode("utf-8")
        return hashlib.sha256(material).hexdigest()


ledger = ApprovalLedger()


__all__ = [
    "ApprovalLedger",
    "ApprovalRecord",
    "ApprovalTokenInvalid",
    "LedgerError",
    "OperationNotApprovable",
    "OperationPayloadTampered",
    "UnknownOperation",
    "WriteLog",
    "WriteRecord",
    "ledger",
]
