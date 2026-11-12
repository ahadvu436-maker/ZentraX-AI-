"""Ingest rows from any DB-API 2.0 database (SQLite built in) as documents."""
from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Union

from . import BaseSource, RawDocument, make_doc_id

logger = logging.getLogger(__name__)

__all__ = ["DatabaseSource", "QuerySpec"]

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|truncate|"
    r"grant|revoke|merge|exec|execute|vacuum|reindex)\b",
    re.IGNORECASE,
)


def _assert_read_only(sql: str) -> None:
    """Reject anything that is not a single SELECT / WITH statement.

    This is defence in depth. Also connect with a read-only database user.
    """
    stripped = sql.strip().rstrip(";").strip()
    if not re.match(r"(?is)^(select|with)\b", stripped):
        raise ValueError("Only SELECT/WITH queries are allowed")
    if ";" in stripped:
        raise ValueError("Multiple statements are not allowed")
    match = _FORBIDDEN.search(stripped)
    if match:
        raise ValueError(f"Forbidden keyword in query: {match.group(0)!r}")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


@dataclass
class QuerySpec:
    """Describes how rows of one query map to documents.

    ``sql`` may use the driver's parameter placeholders together with ``params``.
    """

    name: str
    sql: str
    id_column: str
    text_columns: Sequence[str]
    metadata_columns: Sequence[str] = field(default_factory=tuple)
    title_column: Optional[str] = None
    params: Sequence[Any] = field(default_factory=tuple)


class DatabaseSource(BaseSource):
    """Stream rows as documents, one document per row.

    Provide either ``connection_factory`` (any DB-API connect callable) or
    ``sqlite_path`` (opened read-only).
    """

    source_type = "database"

    def __init__(
        self,
        queries: Sequence[QuerySpec],
        *,
        connection_factory: Optional[Callable[[], Any]] = None,
        sqlite_path: Optional[Union[str, Path]] = None,
        batch_size: int = 500,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name or "database")
        if (connection_factory is None) == (sqlite_path is None):
            raise ValueError("Provide exactly one of connection_factory or sqlite_path")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        for spec in queries:
            _assert_read_only(spec.sql)
            if not spec.text_columns:
                raise ValueError(f"Query {spec.name!r} needs at least one text column")
        self.queries = list(queries)
        self.batch_size = batch_size
        if sqlite_path is not None:
            uri = Path(sqlite_path).expanduser().resolve().as_uri() + "?mode=ro"
            connection_factory = lambda: sqlite3.connect(uri, uri=True)  # noqa: E731
        self._factory = connection_factory

    def health_check(self) -> bool:
        try:
            conn = self._factory()  # type: ignore[misc]
        except Exception:
            return False
        try:
            return True
        finally:
            conn.close()

    def fetch(self) -> Iterator[RawDocument]:
        conn = self._factory()  # type: ignore[misc]
        try:
            for spec in self.queries:
                try:
                    yield from self._run(conn, spec)
                except Exception as exc:  # keep going with the next query
                    logger.error("Query %r failed: %s", spec.name, exc)
        finally:
            conn.close()

    def _run(self, conn: Any, spec: QuerySpec) -> Iterator[RawDocument]:
        cursor = conn.cursor()
        try:
            cursor.execute(spec.sql, tuple(spec.params))
            columns: List[str] = [d[0] for d in cursor.description or []]
            wanted = [spec.id_column, *spec.text_columns, *spec.metadata_columns]
            if spec.title_column:
                wanted.append(spec.title_column)
            missing = [c for c in wanted if c not in columns]
            if missing:
                raise ValueError(f"Columns missing from result: {missing}")
            index = {c: i for i, c in enumerate(columns)}
            while True:
                rows = cursor.fetchmany(self.batch_size)
                if not rows:
                    break
                for row in rows:
                    doc = self._row_to_doc(spec, row, index)
                    if doc is not None:
                        yield doc
        finally:
            try:
                cursor.close()
            except Exception:
                pass

    def _row_to_doc(
        self, spec: QuerySpec, row: Sequence[Any], index: Dict[str, int]
    ) -> Optional[RawDocument]:
        parts = []
        for col in spec.text_columns:
            value = _jsonable(row[index[col]])
            if value is not None and str(value).strip():
                parts.append(f"{col}: {value}")
        if not parts:
            return None
        row_id = _jsonable(row[index[spec.id_column]])
        uri = f"db://{self.name}/{spec.name}/{row_id}"
        title = (
            str(_jsonable(row[index[spec.title_column]]))
            if spec.title_column
            else f"{spec.name} #{row_id}"
        )
        metadata: Dict[str, Any] = {"query": spec.name, "row_id": row_id}
        for col in spec.metadata_columns:
            metadata[col] = _jsonable(row[index[col]])
        return RawDocument(
            id=make_doc_id(uri),
            uri=uri,
            content="\n".join(parts),
            source_type=self.source_type,
            title=title,
            metadata=metadata,
        )
