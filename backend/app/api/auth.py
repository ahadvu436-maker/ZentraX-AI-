"""
ZentraX AI — Auth API Router
===============================
Registration and login endpoints.

Login follows the standard OAuth2 "password" flow via
`OAuth2PasswordRequestForm` (form-encoded `username` + `password`) so it
plugs directly into FastAPI's built-in OAuth2 tooling and the Swagger UI
"Authorize" button — `username` is treated as the user's email.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
    TokenError,
    TokenExpiredError,
    TokenType,
)
from app.database.session import get_db_session as get_db
from app.models.user import User
from app.schemas.auth import (
    RefreshTokenRequest,
    TokenResponse,
    UserRegisterRequest,
    UserResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
)
async def register(
    payload: UserRegisterRequest,
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Create a new user account.

    - Rejects registration if the email is already in use (409).
    - Stores only a bcrypt hash of the password — the plaintext password
      is never persisted or logged.
    """
    existing = await db.execute(select(User).where(User.email == payload.email))
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    user = User(
        email=payload.email,
        hashed_password=hash_password(payload.password),
        full_name=payload.full_name,
    )
    db.add(user)

    # Flush (not commit) to surface DB-level errors (e.g. a race on the
    # unique email constraint) here, with a clean HTTP response, rather
    # than as an unhandled exception when get_db's dependency commits.
    try:
        await db.flush()
        await db.refresh(user)
    except Exception:
        await db.rollback()
        logger.warning("Registration failed for email=%s due to a database error.", payload.email)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    logger.info("New user registered: user_id=%s", user.id)
    return user


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Authenticate and receive an access + refresh token pair",
)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """
    Authenticate with email (as `username`) and password.

    Returns a short-lived access token and a long-lived refresh token.
    Uses a generic error message for both "no such user" and "wrong
    password" so the endpoint doesn't leak which emails are registered.
    """
    result = await db.execute(select(User).where(User.email == form_data.username))
    user = result.scalar_one_or_none()

    if user is None or not verify_password(form_data.password, user.hashed_password or ""):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This account has been deactivated.",
        )

    access_token = create_access_token(subject=user.id)
    refresh_token = create_refresh_token(subject=user.id)

    logger.info("User logged in: user_id=%s", user.id)
    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Exchange a valid refresh token for a new access + refresh token pair",
)
async def refresh_access_token(
    payload: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """
    Issue a new token pair from a valid, unexpired refresh token.

    Rejects the request if the token is expired, malformed, not a
    'refresh'-type token, or if the underlying user no longer exists or
    has been deactivated since the token was issued.
    """
    try:
        claims = decode_token(payload.refresh_token, expected_type=TokenType.REFRESH)
    except TokenExpiredError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has expired. Please log in again.",
        )
    except TokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token.",
        )

    result = await db.execute(select(User).where(User.id == claims.sub))
    user = result.scalar_one_or_none()

    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account is no longer available.",
        )

    return TokenResponse(
        access_token=create_access_token(subject=user.id),
        refresh_token=create_refresh_token(subject=user.id),
    )