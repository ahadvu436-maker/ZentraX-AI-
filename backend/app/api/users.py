"""
ZentraX AI — Users API Router
================================
Endpoints for the authenticated user to view and update their own profile.
Every route is scoped to the caller's own account via `get_current_user` —
there is no path parameter for user id here by design, which makes it
structurally impossible for this router to expose or modify another
user's data (see `app.api.admin` — not yet written — for any future
admin-scoped user management).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.database.session import get_db_session as get_db
from app.models.user import User
from app.schemas.auth import UserResponse
from app.schemas.user import UserUpdateRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/users", tags=["Users"])


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get the current authenticated user's profile",
)
async def read_current_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Return the profile of whichever user the bearer token belongs to.

    No database query needed here beyond what `get_current_user` already
    performed — the dependency has already loaded and validated the user.
    """
    return current_user


@router.patch(
    "/me",
    response_model=UserResponse,
    summary="Update the current authenticated user's profile",
)
async def update_current_user(
    payload: UserUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Partially update the caller's own profile.

    Only fields explicitly present in the request body are changed
    (`exclude_unset=True`) — a client sending `{"full_name": "Jane"}` will
    NOT accidentally clear the email field, for example.
    """
    update_fields = payload.model_dump(exclude_unset=True)

    if not update_fields:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No fields provided to update.",
        )

    if "email" in update_fields and update_fields["email"] != current_user.email:
        existing = await db.execute(
            select(User).where(
                User.email == update_fields["email"],
                User.id != current_user.id,
            )
        )
        if existing.scalar_one_or_none() is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This email is already in use by another account.",
            )
        # Changing the login email is a security-relevant action; in a
        # fuller implementation this would set is_verified=False and
        # trigger a re-verification email rather than applying immediately.
        current_user.is_verified = False

    for field_name, value in update_fields.items():
        setattr(current_user, field_name, value)

    try:
        await db.flush()
        await db.refresh(current_user)
    except Exception:
        await db.rollback()
        logger.warning(
            "Profile update failed for user_id=%s due to a database error.",
            current_user.id,
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Could not update profile — the email may already be in use.",
        )

    logger.info("Profile updated: user_id=%s", current_user.id)
    return current_user