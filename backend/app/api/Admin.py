"""
ZentraX AI — Admin API Router
================================
Administrative endpoints: user listing/lookup, role & status management,
and coarse system metrics.

Every route in this router depends on `get_current_superuser`
(app.api.deps) — not `get_current_user`. That dependency already rejects
any non-admin caller with 403 before a route body ever runs, so RBAC
enforcement lives in one place (deps.py) rather than being re-implemented
per endpoint here.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_superuser
from app.database.session import get_db_session as get_db
from app.models.conversation import Conversation
from app.models.documents import Document
from app.models.messages import Message
from app.models.user import User
from app.schemas.admin import (
    AdminUserListResponse,
    AdminUserResponse,
    AdminUserUpdateRequest,
    SystemMetricsResponse,
)

logger = logging.getLogger(__name__)

# Every route below runs `get_current_superuser` via this router-level
# dependency, in addition to (redundantly, deliberately) declaring it on
# each route's own Depends() for readability/self-documentation. FastAPI
# resolves shared dependencies once per request, so this isn't wasted work.
router = APIRouter(
    prefix="/admin",
    tags=["Admin"],
    dependencies=[Depends(get_current_superuser)],
)


@router.get(
    "/users",
    response_model=AdminUserListResponse,
    summary="List all users (admin only)",
)
async def list_users(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    search: str | None = Query(
        default=None, description="Case-insensitive substring match on email."
    ),
    db: AsyncSession = Depends(get_db),
) -> AdminUserListResponse:
    """Paginated list of every user on the platform, optionally filtered by email."""
    base_query = select(User)
    count_query = select(func.count()).select_from(User)

    if search:
        pattern = f"%{search}%"
        base_query = base_query.where(User.email.ilike(pattern))
        count_query = count_query.where(User.email.ilike(pattern))

    total = (await db.execute(count_query)).scalar_one()

    result = await db.execute(
        base_query.order_by(User.created_at.desc()).offset(offset).limit(limit)
    )
    users = result.scalars().all()

    return AdminUserListResponse(
        users=[AdminUserResponse.model_validate(u) for u in users],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/users/{user_id}",
    response_model=AdminUserResponse,
    summary="Get a specific user by ID (admin only)",
)
async def get_user(
    user_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> User:
    user = await db.get(User, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found."
        )
    return user


@router.patch(
    "/users/{user_id}",
    response_model=AdminUserResponse,
    summary="Update a user's role/status flags (admin only)",
)
async def update_user_status(
    user_id: uuid.UUID,
    payload: AdminUserUpdateRequest,
    current_admin: User = Depends(get_current_superuser),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Update `is_active` / `is_verified` / `is_superuser` on any user.

    Refuses to let an admin revoke their OWN `is_superuser` or `is_active`
    flag through this endpoint — self-demotion/self-deactivation here could
    leave the platform with zero admins with no recovery path short of a
    direct DB edit. (An admin can still be demoted/deactivated by a
    *different* admin.)
    """
    target_user = await db.get(User, user_id)
    if target_user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found."
        )

    update_fields = payload.model_dump(exclude_unset=True)
    if not update_fields:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No fields provided to update.",
        )

    if target_user.id == current_admin.id:
        if update_fields.get("is_superuser") is False:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="You cannot revoke your own administrator privileges.",
            )
        if update_fields.get("is_active") is False:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="You cannot deactivate your own account.",
            )

    for field_name, value in update_fields.items():
        setattr(target_user, field_name, value)

    await db.flush()
    await db.refresh(target_user)

    logger.info(
        "Admin action: admin_id=%s updated user_id=%s fields=%s",
        current_admin.id,
        target_user.id,
        list(update_fields.keys()),
    )

    return target_user


@router.get(
    "/metrics",
    response_model=SystemMetricsResponse,
    summary="Get platform-wide usage metrics (admin only)",
)
async def get_system_metrics(
    db: AsyncSession = Depends(get_db),
) -> SystemMetricsResponse:
    """
    Coarse counts across core tables. A point-in-time snapshot suitable
    for an admin dashboard — not intended for high-frequency polling; add
    caching (e.g. short-TTL Redis) if this becomes a hot path.
    """
    total_users = (await db.execute(select(func.count()).select_from(User))).scalar_one()
    active_users = (
        await db.execute(
            select(func.count()).select_from(User).where(User.is_active.is_(True))
        )
    ).scalar_one()
    total_conversations = (
        await db.execute(select(func.count()).select_from(Conversation))
    ).scalar_one()
    total_messages = (
        await db.execute(select(func.count()).select_from(Message))
    ).scalar_one()
    total_documents = (
        await db.execute(select(func.count()).select_from(Document))
    ).scalar_one()

    return SystemMetricsResponse(
        total_users=total_users,
        active_users=active_users,
        total_conversations=total_conversations,
        total_messages=total_messages,
        total_documents=total_documents,
        generated_at=datetime.now(timezone.utc),
    )