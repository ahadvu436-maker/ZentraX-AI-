"""
ZentraX AI — Document Model
==============================
SQLAlchemy ORM model for the `documents` table.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.session import Base

if TYPE_CHECKING:
    from app.models.user import User


class Document(Base):
    """
    A file uploaded by a user (e.g. for retrieval-augmented generation).

    NOTE: The Python attribute is named `doc_metadata`, not `metadata` —
    `metadata` is a reserved name on SQLAlchemy's `DeclarativeBase` (it holds
    the table/schema registry), so using it as a column attribute would shadow
    that and break the model. The actual database column is still named
    `metadata`, via the explicit `name="metadata"` argument below, so the
    schema matches your spec exactly.
    """

    __tablename__ = "documents"

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

    filename: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
    )

    file_url: Mapped[str] = mapped_column(
        String(2048),
        nullable=False,
        doc="Location of the stored file, e.g. an S3/GCS object URL or path.",
    )

    doc_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
        doc="Arbitrary document metadata (e.g. mime_type, page_count, "
        "checksum, ingestion status, source system).",
    )

    # NOTE: `created_at` (and `updated_at`) are inherited from `Base` in
    # app.database.session — not redeclared here.

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------
    user: Mapped["User"] = relationship(
        "User",
        back_populates="documents",
        lazy="joined",
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Document id={self.id} user_id={self.user_id} filename={self.filename!r}>"
