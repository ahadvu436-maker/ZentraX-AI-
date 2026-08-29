"""
ZentraX AI — Database Session Management
==========================================
Async SQLAlchemy engine, session factory, and declarative base.

This module is the single source of truth for DB connectivity. All models
should inherit from `Base` defined here, and all request-scoped DB access
should go through `get_db_session()` (a FastAPI dependency) or, outside of
request scope, `session_scope()` (an async context manager).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import NullPool, QueuePool

from app.config.settings import settings

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Naming convention — ensures Alembic autogenerate produces consistent,
# predictable constraint/index names across environments.
# ----------------------------------------------------------------------
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=NAMING_CONVENTION)


class Base(DeclarativeBase):
    """
    Shared declarative base for all ORM models.

    Provides common audit columns (`created_at`, `updated_at`) so individual
    models don't need to redeclare them. Models should inherit directly from
    this class:

        class Document(Base):
            __tablename__ = "documents"
            id: Mapped[uuid.UUID] = mapped_column(primary_key=True, ...)
    """

    metadata = metadata

    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        pk_cols = [c.name for c in self.__table__.primary_key.columns]
        pk_vals = ", ".join(f"{c}={getattr(self, c, None)!r}" for c in pk_cols)
        return f"<{self.__class__.__name__} {pk_vals}>"


# ----------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------
def _build_engine() -> AsyncEngine:
    """
    Build the async engine.

    Uses a real pool (QueuePool-equivalent for async) in normal operation,
    and NullPool under TEST environment to avoid cross-test connection
    reuse / event-loop issues with pytest-asyncio.
    """
    is_test = settings.ENVIRONMENT.value == "test"

    engine_kwargs: dict = {
        "echo": settings.DB_ECHO_SQL,
        "future": True,
        "pool_pre_ping": True,  # detects stale connections before use
    }

    if is_test:
        engine_kwargs["poolclass"] = NullPool
    else:
        engine_kwargs.update(
            pool_size=settings.DB_POOL_SIZE,
            max_overflow=settings.DB_MAX_OVERFLOW,
            pool_timeout=settings.DB_POOL_TIMEOUT_SECONDS,
            pool_recycle=settings.DB_POOL_RECYCLE_SECONDS,
        )

    return create_async_engine(settings.sqlalchemy_database_uri, **engine_kwargs)


engine: AsyncEngine = _build_engine()

# ----------------------------------------------------------------------
# Session factory
# ----------------------------------------------------------------------
AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,  # allow using ORM objects after commit (e.g. in response schemas)
    autoflush=False,
    autocommit=False,
)


# ----------------------------------------------------------------------
# FastAPI dependency
# ----------------------------------------------------------------------
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields a request-scoped `AsyncSession`.

    Commits on clean exit, rolls back on exception, always closes.
    Usage:

        @router.get("/items")
        async def list_items(db: AsyncSession = Depends(get_db_session)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ----------------------------------------------------------------------
# Context manager for non-request-scoped DB access
# (background jobs, scripts, startup hooks, etc.)
# ----------------------------------------------------------------------
@asynccontextmanager
async def session_scope() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for DB access outside FastAPI's dependency system.

    Usage:

        async with session_scope() as db:
            result = await db.execute(select(Document))
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ----------------------------------------------------------------------
# Lifespan hooks
# ----------------------------------------------------------------------
async def init_db() -> None:
    """
    Verify DB connectivity and ensure the pgvector extension is enabled.

    Called once at application startup (see app lifespan handler). Table
    creation/migrations are intentionally NOT performed here — use Alembic
    migrations for schema management in all environments.
    """
    from sqlalchemy import text

    async with engine.begin() as conn:
        await conn.execute(
            text(f'CREATE EXTENSION IF NOT EXISTS "{settings.PGVECTOR_EXTENSION}"')
        )
        logger.info(
            "Database connectivity verified; '%s' extension ensured.",
            settings.PGVECTOR_EXTENSION,
        )


async def close_db() -> None:
    """Dispose of the engine's connection pool. Call on application shutdown."""
    await engine.dispose()
    logger.info("Database engine disposed.")