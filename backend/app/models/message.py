"""
ZentraX AI — Message Model
=============================
SQLAlchemy ORM model for the `messages` table.
"""

from __future__ import annotations

import enum
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Integer, Text
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.session import Base

if TYPE_CHECKING:
    from app.models.conversation import Conversation


class SenderType(str, enum.Enum):
    """Who authored a given message in a conversation."""

    USER = "user"
    AI = "ai"


class Message(Base):
    """
    A single message (turn) within a `Conversation`.

    `token_usage` stores the token count consumed for generating/processing
    this specific message (e.g. completion tokens for an AI message). It is
    nullable since user-authored messages may not always have a meaningful
    token count recorded at write time.
    """

    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=None,
    )

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    sender_type: Mapped[SenderType] = mapped_column(
        SAEnum(
            SenderType,
            name="sender_type_enum",
            native_enum=True,
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        nullable=False,
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    token_usage: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        doc="Number of tokens consumed by this message (prompt or completion, "
        "depending on sender_type). Null when not tracked.",
    )

    # NOTE: `created_at` (and `updated_at`) are inherited from `Base` in
    # app.database.session — not redeclared here. Messages are effectively
    # immutable, but `updated_at` remains available (e.g. for moderation edits).

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------
    conversation: Mapped["Conversation"] = relationship(
        "Conversation",
        back_populates="messages",
        lazy="joined",
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<Message id={self.id} conversation_id={self.conversation_id} "
            f"sender_type={self.sender_type.value}>"
        )