"""
ZentraX AI — User Model
=========================
SQLAlchemy ORM model for the `users` table.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.session import Base

if TYPE_CHECKING:
    from app.models.conversation import Conversation


class User(Base):
    """
    A registered ZentraX AI user / account holder.

    Notes:
    - `hashed_password` is nullable to support future SSO/OAuth-only accounts
      that never set a local password.
    - `email` is the natural login identifier and is uniquely indexed.
    """

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=None,
    )

    email: Mapped[str] = mapped_column(
        String(320),  # RFC 5321 max email length
        unique=True,
        index=True,
        nullable=False,
    )

    hashed_password: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    full_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    is_superuser: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------
    conversations: Mapped[list["Conversation"]] = relationship(
        "Conversation",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="selectin",
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<User id={self.id} email={self.email!r}>"