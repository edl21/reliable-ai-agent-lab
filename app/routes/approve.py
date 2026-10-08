"""POST /approve — human/application approval boundary."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from adapter.runtime import current_fault
from app.errors import OperationTampered
from app.ledger import (
    OperationNotApprovable,
    OperationPayloadTampered,
    UnknownOperation,
    ledger,
)
from app.schemas import ApproveRequest, ApproveResponse
from traces.schema import JSONLWriter, TraceEvent


router = APIRouter()
_TRACE_ROOT = Path("traces/task2")


@router.post("/approve", response_model=ApproveResponse)
def approve(
    request: ApproveRequest,
    trace_check_name: str = Query("adhoc"),
) -> ApproveResponse:
    """Approve, reject, or replay a pending operation.

    This is deliberately not exposed as a model tool. The model can
    propose ``save_guide``; only this separate route can authorize the
    write.
    """
    run_id = uuid.uuid4().hex
    trace_root = (
        Path("traces/task3")
        if trace_check_name.startswith("R")
        else _TRACE_ROOT
    )
    trace_path = trace_root / f"{trace_check_name}_{request.operation_id}.jsonl"
    trace_path.unlink(missing_ok=True)
    writer = JSONLWriter()

    try:
        record, replayed = ledger.approve(
            operation_id=request.operation_id,
            approve=request.approve,
            payload_check=request.payload_check,
        )
    except UnknownOperation as exc:
        _write_approval_trace(
            writer,
            trace_path,
            run_id,
            request.operation_id,
            "rejected",
            {"reason": "unknown_operation"},
        )
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "code": "unknown_operation",
                    "message": str(exc),
                }
            },
        ) from exc
    except OperationPayloadTampered as exc:
        _write_approval_trace(
            writer,
            trace_path,
            run_id,
            request.operation_id,
            "rejected",
            {"reason": "payload_hash_mismatch"},
        )
        raise OperationTampered(request.operation_id) from exc
    except OperationNotApprovable as exc:
        _write_approval_trace(
            writer,
            trace_path,
            run_id,
            request.operation_id,
            "rejected",
            {"reason": "operation_not_approvable"},
        )
        raise HTTPException(
            status_code=409,
            detail={
                "error": {
                    "code": "operation_not_approvable",
                    "message": str(exc),
                }
            },
        ) from exc

    status = "already_saved" if replayed else (
        "saved" if record.state == "saved" else "rejected"
    )
    _write_approval_trace(
        writer,
        trace_path,
        run_id,
        request.operation_id,
        status,
        {
            "payload_hash": record.payload_hash,
            "artifact_id": record.artifact_id,
            "write_count": ledger.write_log.count(request.operation_id),
            "replayed": replayed,
            "approval_record_state": record.state,
        },
    )
    if not replayed:
        fault = current_fault()
        if fault is not None:
            fault.after_approval_write(request.operation_id)
    return ApproveResponse(
        operation_id=record.operation_id,
        status=status,
        artifact_id=record.artifact_id,
    )


def _write_approval_trace(
    writer: JSONLWriter,
    path: Path,
    run_id: str,
    operation_id: str,
    outcome: str,
    notes: dict[str, object],
) -> None:
    writer.append(
        path,
        TraceEvent(
            request_id=operation_id,
            run_id=run_id,
            task="t2",
            operation="approve",
            outcome="saved" if outcome in {"saved", "already_saved"} else "rejected",
            recovery_decision="replay_ledger"
            if outcome == "already_saved"
            else "none",
            notes={
                "operation_id": operation_id,
                "approval_outcome": outcome,
                **notes,
            },
        ),
    )


__all__ = ["router"]
