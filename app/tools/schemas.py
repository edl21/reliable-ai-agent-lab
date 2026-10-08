"""Strict schemas and OpenAI tool declarations for Task 2."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SearchPassagesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=2_000)


class FetchPassageArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(..., min_length=1, max_length=128)


class GuideCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    source_filename: str
    chapter: int
    pdf_pages: list[int]
    excerpt: str


class SaveGuideArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=1, max_length=20_000)
    citations: list[GuideCitation] = Field(..., min_length=1)


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_passages",
            "description": (
                "Search permitted book passages. Results are filtered "
                "to the request's inclusive chapter limit."
            ),
            "parameters": SearchPassagesArgs.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_passage",
            "description": (
                "Fetch one passage by chunk ID. The chunk must be within "
                "the request's inclusive chapter limit."
            ),
            "parameters": FetchPassageArgs.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_guide",
            "description": (
                "Create a pending guide draft. This tool never writes an "
                "artifact and cannot grant approval."
            ),
            "parameters": SaveGuideArgs.model_json_schema(),
        },
    },
]


__all__ = [
    "FetchPassageArgs",
    "GuideCitation",
    "SaveGuideArgs",
    "SearchPassagesArgs",
    "TOOL_SCHEMAS",
]
