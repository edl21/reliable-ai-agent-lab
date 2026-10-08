"""Structured tool dispatcher for model-driven Task 2 calls."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import ValidationError

from app.tools.context import ToolContext
from app.tools.fetch_passage import fetch_passage
from app.tools.save_guide import save_guide
from app.tools.schemas import (
    FetchPassageArgs,
    SaveGuideArgs,
    SearchPassagesArgs,
)
from app.tools.search_passages import search_passages


@dataclass
class DispatchResult:
    """The dispatcher result is always safe to send back as a tool-role
    message. Invalid calls have ``ok=False`` and never execute."""

    name: str
    call_id: str
    ok: bool
    payload: dict[str, Any]
    pending: bool = False

    def content(self) -> str:
        return json.dumps(self.payload, separators=(",", ":"), ensure_ascii=False)


class ToolDispatcher:
    """Validates name and arguments before calling any implementation."""

    def __init__(self, *, context: ToolContext) -> None:
        self.context = context
        self.invocations: list[dict[str, Any]] = []

    def dispatch(
        self,
        *,
        name: str,
        arguments: Any,
        call_id: str,
    ) -> DispatchResult:
        if name not in _HANDLERS:
            return self._error(
                name=name,
                call_id=call_id,
                code="unknown_tool",
                detail=f"unknown tool name: {name}",
            )
        if not isinstance(arguments, dict):
            return self._error(
                name=name,
                call_id=call_id,
                code="invalid_arguments",
                detail="tool arguments must be a JSON object",
            )

        try:
            if name == "search_passages":
                args = SearchPassagesArgs.model_validate(arguments)
                result = search_passages(args.query, context=self.context)
            elif name == "fetch_passage":
                args = FetchPassageArgs.model_validate(arguments)
                result = fetch_passage(args.chunk_id, context=self.context)
            else:
                args = SaveGuideArgs.model_validate(arguments)
                result = save_guide(
                    args.title,
                    args.content,
                    args.citations,
                    context=self.context,
                )
        except ValidationError as exc:
            return self._error(
                name=name,
                call_id=call_id,
                code="invalid_arguments",
                detail=exc.errors(include_url=False),
            )
        except Exception as exc:  # tool failures become model-correctable data
            return self._error(
                name=name,
                call_id=call_id,
                code="tool_execution_error",
                detail=str(exc),
            )

        self.invocations.append(
            {"name": name, "call_id": call_id, "arguments": arguments}
        )
        return DispatchResult(
            name=name,
            call_id=call_id,
            ok=True,
            payload=result,
            pending=(name == "save_guide"),
        )

    @staticmethod
    def _error(
        *,
        name: str,
        call_id: str,
        code: str,
        detail: Any,
    ) -> DispatchResult:
        return DispatchResult(
            name=name,
            call_id=call_id,
            ok=False,
            payload={
                "error": {
                    "code": "tool_validation_error"
                    if code == "invalid_arguments"
                    else code,
                    "tool": name,
                    "detail": detail,
                }
            },
        )


_HANDLERS: dict[str, Callable[..., Any]] = {
    "search_passages": search_passages,
    "fetch_passage": fetch_passage,
    "save_guide": save_guide,
}


__all__ = ["DispatchResult", "ToolDispatcher"]
