"""
ZentraX AI — Conversation Model
==================================
SQLAlchemy ORM model for the `conversations` table.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.session import Base

if TYPE_CHECKING:
    from app.models.user import User


class Conversation(Base):
    """
    A single conversation/thread belonging to a user.

    Kept intentionally lean at this stage — individual chat turns are
    expected to live in a separate `messages` table (not yet defined) with
    a FK back to `Conversation.id`, so this model does not embed message
    content directly.
    """

    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=None,
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    title: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    summary: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="Optional rolling/compacted summary of the conversation, used for "
        "long-context retrieval instead of replaying full message history.",
    )

    is_archived: Mapped[bool] = mapped_column(
        default=False,
        nullable=False,
    )

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------
    user: Mapped["User"] = relationship(
        "User",
        back_populates="conversations",
        lazy="joined",
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Conversation id={self.id} user_id={self.user_id} title={self.title!r}>"