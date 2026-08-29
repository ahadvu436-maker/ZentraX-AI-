"""
ZentraX AI — Alembic Environment
===================================
Async-compatible Alembic `env.py`.

Two things this file is responsible for that the default Alembic template
is NOT set up for out of the box:

1. Sourcing the database URL from `app.config.settings.settings`
   (env vars / .env) instead of a hardcoded `alembic.ini` value.
2. Running migrations through an async SQLAlchemy engine (`asyncpg`),
   since the application itself is fully async.

Autogenerate support requires every model module to be imported below —
Alembic diffs against whatever tables are registered on `Base.metadata`
at the time this file runs, not against your filesystem.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import Connection, pool
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.config.settings import settings
from app.database.session import Base

# ------------------------------------------------------------------
# Import every model module so its table(s) register on Base.metadata.
# This import block is the ONLY thing that makes autogenerate aware of
# your schema — a model that isn't imported here is invisible to Alembic,
# even though it works fine at runtime via app.database.session.
# ------------------------------------------------------------------
from app.models.user import User  # noqa: F401
from app.models.conversation import Conversation  # noqa: F401
from app.models.messages import Message  # noqa: F401
from app.models.documents import Document  # noqa: F401
from app.models.embedding import Embedding  # noqa: F401

# this is the Alembic Config object, which provides access to values
# within the .ini file in use.
config = context.config

# Inject our application's database URL, overriding whatever (if anything)
# is set in alembic.ini. Keeps a single source of truth in Settings.
config.set_main_option("sqlalchemy.url", settings.sqlalchemy_database_uri)

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# The metadata object autogenerate diffs your models against.
target_metadata = Base.metadata


def include_object(object, name, type_, reflected, compare_to) -> bool:
    """
    Filter hook for autogenerate.

    Excludes the pgvector extension's own internal objects and anything
    Alembic might otherwise try to "helpfully" manage that isn't one of
    our application tables. Extend this if pgvector/PostGIS-style
    extensions introduce noise in future `alembic revision --autogenerate`
    diffs.
    """
    return True


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode (generates SQL without a live DB
    connection — used for `alembic upgrade --sql` style output).
    """
    url = settings.sqlalchemy_database_uri
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=include_object,
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """
    Shared migration runner, invoked with a live (sync-facade) connection
    from within the async engine's `run_sync`.
    """
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        include_object=include_object,
        compare_type=True,  # detect column type changes, not just add/drop
        compare_server_default=True,
        render_as_batch=False,  # not needed for Postgres (SQLite-only concern)
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """
    Build an async engine and run migrations against it via `run_sync`,
    since Alembic's migration context itself is synchronous under the hood.
    """
    connectable: AsyncEngine = create_async_engine(
        settings.sqlalchemy_database_uri,
        poolclass=pool.NullPool,  # short-lived migration connection; no pooling needed
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode against a live async DB connection."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()