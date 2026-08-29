"""
ZentraX AI — Embedding Model
===============================
SQLAlchemy ORM model for the `embeddings` table, backed by pgvector.

Each row represents one chunk of a `Document`, together with the vector
representation of that chunk's text — the unit retrieval operates on for
RAG (retrieval-augmented generation).

Requires the `pgvector` Python package (`pip install pgvector`) and the
Postgres `vector` extension, which is enabled at startup in
`app.database.session.init_db()`.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, Index, Integer, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.config.settings import settings
from app.database.session import Base

if TYPE_CHECKING:
    from app.models.documents import Document


class Embedding(Base):
    """
    A vector embedding for a single chunk of a source `Document`.

    Documents are split into chunks at ingestion time (chunking strategy
    lives in the ingestion service, not here); each chunk gets its own row
    so retrieval can return precise, citation-sized spans rather than whole
    documents.

    `embedding` dimensionality is driven by `settings.EMBEDDING_DIMENSIONS`
    and MUST match whatever embedding model is configured — mismatched
    dimensions will fail at insert time at the database level.
    """

    __tablename__ = "embeddings"
    __table_args__ = (
        UniqueConstraint(
            "document_id", "chunk_index", name="uq_embeddings_document_chunk"
        ),
        # IVFFlat index for approximate nearest-neighbor search on cosine
        # distance. `lists` should be tuned to roughly sqrt(row_count) once
        # the table has real data; a reasonable starting point is left here.
        # NOTE: IVFFlat must be created via Alembic/raw SQL after the table
        # has data for best results — this Index() declaration documents
        # intent and works for autogenerate, but consider building/rebuilding
        # it manually in production once row counts are known.
        Index(
            "ix_embeddings_embedding_cosine",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_with={"lists": 100},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=None,
    )

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="0-based position of this chunk within the source document, "
        "used for ordering and citation ('chunk 3 of 12').",
    )

    chunk_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="The raw text content of this chunk, kept alongside the vector "
        "so retrieval can return the source text without a second lookup.",
    )

    token_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        doc="Number of tokens in chunk_text at embedding time, for context-"
        "window budgeting during retrieval.",
    )

    embedding: Mapped[list[float]] = mapped_column(
        Vector(settings.EMBEDDING_DIMENSIONS),
        nullable=False,
    )

    # NOTE: `created_at` / `updated_at` inherited from `Base`.

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------
    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="embeddings",
        lazy="joined",
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<Embedding id={self.id} document_id={self.document_id} "
            f"chunk_index={self.chunk_index}>"
        )