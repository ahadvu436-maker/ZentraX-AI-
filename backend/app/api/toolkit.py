"""
ZentraX AI — Toolkit API Router
==================================
Endpoints for discovering and invoking registered AI toolkit
functions ("tools"/"plugins" in the function-calling sense).

See `app.services.toolkit` for the registry and the security boundary it
enforces: only pre-registered, developer-defined callables can execute —
this router never evaluates arbitrary code from a request.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_current_user
from app.models.user import User
from app.schemas.toolkit import (
    ToolExecuteRequest,
    ToolExecuteResponse,
    ToolInfo,
    ToolListResponse,
)
from app.services.toolkit import (
    ToolAuthorizationError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolValidationError,
    execute_tool,
    list_tools,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/toolkit", tags=["Toolkit"])


@router.get(
    "/tools",
    response_model=ToolListResponse,
    summary="List available toolkit functions",
)
async def get_available_tools(
    current_user: User = Depends(get_current_user),
) -> ToolListResponse:
    """
    List every tool the current user is permitted to see.

    Admin-only tools are still listed (with `requires_admin: true`) so
    clients can render them as disabled/locked rather than having them
    silently disappear — actual enforcement happens at execution time,
    not at discovery time.
    """
    tools = [
        ToolInfo(
            name=t.name,
            description=t.description,
            parameters_schema=t.parameters_model.model_json_schema(),
            requires_admin=t.requires_admin,
        )
        for t in list_tools()
    ]
    return ToolListResponse(tools=tools)


@router.post(
    "/execute",
    response_model=ToolExecuteResponse,
    summary="Execute a registered toolkit function",
)
async def run_tool(
    payload: ToolExecuteRequest,
    current_user: User = Depends(get_current_user),
) -> ToolExecuteResponse:
    """
    Validate and execute a single named tool with the given parameters.

    Parameter validation failures return 422 with per-field error detail
    (mirroring FastAPI's native validation error shape). Unknown tool
    names return 404. Admin-only tools return 403 for non-admin callers.
    Handler exceptions are caught and reported as a structured failure in
    the response body (`success: false`) rather than a raw 500, so a
    single flaky tool doesn't look like a platform outage to the caller.
    """
    try:
        result, duration_ms = await execute_tool(
            name=payload.tool_name,
            raw_parameters=payload.parameters,
            current_user=current_user,
        )
    except ToolNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except ToolAuthorizationError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    except ToolValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": str(exc), "errors": exc.errors},
        )
    except ToolExecutionError as exc:
        return ToolExecuteResponse(
            tool_name=payload.tool_name,
            success=False,
            error=str(exc),
            executed_at=datetime.now(timezone.utc),
            duration_ms=0.0,
        )

    logger.info(
        "Tool executed: tool=%s user_id=%s duration_ms=%.2f",
        payload.tool_name,
        current_user.id,
        duration_ms,
    )

    return ToolExecuteResponse(
        tool_name=payload.tool_name,
        success=True,
        result=result,
        executed_at=datetime.now(timezone.utc),
        duration_ms=duration_ms,
    )