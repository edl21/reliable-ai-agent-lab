"""FastAPI application wiring for Reliable AI Agent Lab.

The app deliberately does not instantiate a model SDK client here.
Every model call goes through the shared adapter, keeping retry,
recovery, and tracing behavior consistent across workflows.
"""

from __future__ import annotations

from fastapi import FastAPI

from app.routes.answer import router as answer_router
from app.routes.approve import router as approve_router
from app.routes.dashboard_ui import router as dashboard_router
from app.routes.guide import router as guide_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Reliable AI Agent Lab",
        version="1.0.0",
        description=(
            "Grounded question answering, human-approved tools, "
            "resilient model execution, and constrained repair."
        ),
    )
    app.include_router(answer_router)
    app.include_router(guide_router)
    app.include_router(approve_router)
    app.include_router(dashboard_router)
    return app


app = create_app()
