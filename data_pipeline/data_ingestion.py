"""Data ingestion: pluggable sources, validation of the raw frame, retry-aware step.

Design
------
* A :class:`DataSource` knows how to ``read`` (and optionally stream in
  batches) one kind of input: CSV, JSON / JSON-lines, Parquet, SQLite / SQL,
  or an in-memory ``DataFrame``.
* :class:`DataIngestor` wraps a source and applies the *structural* checks
  that every pipeline needs: error translation, snake_case column names,
  unique column names, required columns and a minimum row count.
* :class:`IngestionStep` plugs an ingestor into a :class:`~Data_pipeline.Pipeline`
  and retries only *transient* I/O failures (never a missing file or a bad
  parse).
"""

from __future__ import annotations

import abc
import logging
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Union

import pandas as pd

from .pipeline_core import (
    PipelineContext,
    PipelineError,
    PipelineStep,
    RetryPolicy,
    utc_now_iso,
)

__all__ = [
    "IngestionError",
    "TransientIngestionError",
    "normalize_column_name",
    "DataSource",
    "CSVSource",
    "JSONSource",
    "ParquetSource",
    "SQLSource",
    "DataFrameSource",
    "DataIngestor",
    "IngestionStep",
]

logger = logging.getLogger(__name__)


class IngestionError(PipelineError):
    """Data could not be read or failed a structural check (not retryable)."""


class TransientIngestionError(IngestionError):
    """Temporary I/O problem (timeouts, locked database ...); safe to retry."""


_NON_ALNUM = re.compile(r"[^0-9a-zA-Z]+")


def normalize_column_name(name: Any) -> str:
    """Convert a header to ``snake_case`` (``"Age (yrs)"`` -> ``"age_yrs"``)."""
    text = _NON_ALNUM.sub("_", str(name).strip()).strip("_").lower()
    return text or "column"


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #
class DataSource(abc.ABC):
    """Something that can produce a DataFrame."""

    @abc.abstractmethod
    def read(self) -> pd.DataFrame:
        """Read the full dataset."""

    def iter_batches(self, batch_size: int) -> Iterator[pd.DataFrame]:
        """Yield the dataset in batches (default: read everything, then slice)."""
        if batch_size < 1:
            raise IngestionError("batch_size must be >= 1")
        frame = self.read()
        for start in range(0, len(frame), batch_size):
            yield frame.iloc[start : start + batch_size]

    @property
    def description(self) -> str:
        return type(self).__name__


class _FileSource(DataSource):
    """Shared path handling: existence, file-ness and optional size limit."""

    def __init__(self, path: Union[str, Path], *, max_bytes: Optional[int] = None) -> None:
        self.path = Path(path).expanduser()
        self.max_bytes = max_bytes

    def _checked_path(self) -> Path:
        resolved = self.path.resolve()
        if not resolved.is_file():
            raise IngestionError(f"File not found: {resolved}")
        if self.max_bytes is not None and resolved.stat().st_size > self.max_bytes:
            raise IngestionError(
                f"{resolved.name} exceeds max_bytes={self.max_bytes} "
                f"({resolved.stat().st_size} bytes)"
            )
        return resolved

    @property
    def description(self) -> str:
        return f"{type(self).__name__}({self.path})"


class CSVSource(_FileSource):
    """CSV / TSV (optionally ``.gz``) file.

    Args:
        path: File path.
        delimiter: Field separator.
        encoding: Text encoding.
        read_kwargs: Extra keyword arguments for :func:`pandas.read_csv`
            (e.g. ``{"parse_dates": ["created_at"], "dtype": {...}}``).
        max_bytes: Reject files larger than this.
    """

    def __init__(
        self,
        path: Union[str, Path],
        *,
        delimiter: str = ",",
        encoding: str = "utf-8",
        read_kwargs: Optional[Mapping[str, Any]] = None,
        max_bytes: Optional[int] = None,
    ) -> None:
        super().__init__(path, max_bytes=max_bytes)
        self.delimiter = delimiter
        self.encoding = encoding
        self.read_kwargs: Dict[str, Any] = dict(read_kwargs or {})

    def read(self) -> pd.DataFrame:
        return pd.read_csv(
            self._checked_path(), sep=self.delimiter, encoding=self.encoding, **self.read_kwargs
        )

    def iter_batches(self, batch_size: int) -> Iterator[pd.DataFrame]:
        if batch_size < 1:
            raise IngestionError("batch_size must be >= 1")
        yield from pd.read_csv(
            self._checked_path(),
            sep=self.delimiter,
            encoding=self.encoding,
            chunksize=batch_size,
            **self.read_kwargs,
        )


class JSONSource(_FileSource):
    """JSON array/records file or JSON-lines (``lines=True``)."""

    def __init__(
        self,
        path: Union[str, Path],
        *,
        lines: bool = False,
        read_kwargs: Optional[Mapping[str, Any]] = None,
        max_bytes: Optional[int] = None,
    ) -> None:
        super().__init__(path, max_bytes=max_bytes)
        self.lines = lines
        self.read_kwargs: Dict[str, Any] = dict(read_kwargs or {})

    def read(self) -> pd.DataFrame:
        return pd.read_json(self._checked_path(), lines=self.lines, **self.read_kwargs)


class ParquetSource(_FileSource):
    """Parquet file (needs ``pyarrow`` or ``fastparquet``)."""

    def __init__(
        self,
        path: Union[str, Path],
        *,
        columns: Optional[Sequence[str]] = None,
        max_bytes: Optional[int] = None,
    ) -> None:
        super().__init__(path, max_bytes=max_bytes)
        self.columns = list(columns) if columns is not None else None

    def read(self) -> pd.DataFrame:
        return pd.read_parquet(self._checked_path(), columns=self.columns)


class SQLSource(DataSource):
    """Result of a SQL query.

    Args:
        connection: Path to a SQLite file (opened **read-only**), or an
            existing DB-API connection / SQLAlchemy connectable.
        query: SQL text. Always pass values through ``params`` - never
            build the query with string formatting.
        params: Query parameters.
    """

    def __init__(
        self,
        connection: Any,
        query: str,
        params: Optional[Union[Sequence[Any], Mapping[str, Any]]] = None,
    ) -> None:
        self.connection = connection
        self.query = query
        self.params = params

    def read(self) -> pd.DataFrame:
        if isinstance(self.connection, (str, Path)):
            db_path = Path(self.connection).expanduser().resolve()
            if not db_path.is_file():
                raise IngestionError(f"Database not found: {db_path}")
            conn = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)
            try:
                return pd.read_sql_query(self.query, conn, params=self.params)
            finally:
                conn.close()
        return pd.read_sql_query(self.query, self.connection, params=self.params)

    @property
    def description(self) -> str:
        target = self.connection if isinstance(self.connection, (str, Path)) else "connection"
        return f"SQLSource({target})"


class DataFrameSource(DataSource):
    """In-memory DataFrame (a defensive copy is returned)."""

    def __init__(self, frame: pd.DataFrame) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise IngestionError(f"Expected DataFrame, got {type(frame).__name__}")
        self._frame = frame

    def read(self) -> pd.DataFrame:
        return self._frame.copy()


# --------------------------------------------------------------------------- #
# Ingestor
# --------------------------------------------------------------------------- #
class DataIngestor:
    """Read from a :class:`DataSource` and enforce structural expectations.

    Args:
        source: Where the data comes from.
        required_columns: Columns that must exist (names are normalised the
            same way as the data when ``normalize_columns`` is on).
        min_rows: Minimum number of rows.
        normalize_columns: Convert headers to ``snake_case``. Refer to columns
            by their normalised names everywhere downstream.
    """

    def __init__(
        self,
        source: DataSource,
        *,
        required_columns: Sequence[str] = (),
        min_rows: int = 1,
        normalize_columns: bool = True,
    ) -> None:
        if not isinstance(source, DataSource):
            raise IngestionError(f"source must be a DataSource, got {type(source).__name__}")
        self.source = source
        self.normalize_columns = normalize_columns
        self.min_rows = min_rows
        self.required_columns = tuple(
            normalize_column_name(c) if normalize_columns else c for c in required_columns
        )

    # -- construction helpers -------------------------------------------- #
    @classmethod
    def from_path(
        cls,
        path: Union[str, Path],
        *,
        ingestor_kwargs: Optional[Mapping[str, Any]] = None,
        **source_kwargs: Any,
    ) -> "DataIngestor":
        """Pick a source from the file extension.

        ``.csv/.tsv/.txt`` (+``.gz``), ``.json``, ``.jsonl/.ndjson``,
        ``.parquet``. SQLite files need an explicit :class:`SQLSource`.
        """
        p = Path(path).expanduser()
        name = p.name.lower()
        if name.endswith(".gz"):
            name = name[:-3]
        suffix = Path(name).suffix
        source: DataSource
        if suffix in (".csv", ".txt"):
            source = CSVSource(p, **source_kwargs)
        elif suffix == ".tsv":
            source_kwargs.setdefault("delimiter", "\t")
            source = CSVSource(p, **source_kwargs)
        elif suffix == ".json":
            source = JSONSource(p, **source_kwargs)
        elif suffix in (".jsonl", ".ndjson"):
            source_kwargs.setdefault("lines", True)
            source = JSONSource(p, **source_kwargs)
        elif suffix in (".parquet", ".pq"):
            source = ParquetSource(p, **source_kwargs)
        else:
            raise IngestionError(
                f"Cannot infer a source for '{p.name}'. Build a DataSource explicitly."
            )
        return cls(source, **dict(ingestor_kwargs or {}))

    # -- public API ------------------------------------------------------- #
    def ingest(self) -> pd.DataFrame:
        """Read the dataset and run the structural checks."""
        started = time.perf_counter()
        frame = self._guarded(self.source.read)
        frame = self._postprocess(frame, check_rows=True)
        logger.info(
            "Ingested %d rows x %d cols from %s in %.3fs",
            len(frame), frame.shape[1], self.source.description, time.perf_counter() - started,
        )
        return frame

    def iter_batches(self, batch_size: int) -> Iterator[pd.DataFrame]:
        """Stream the dataset as checked batches (row minimum is not enforced)."""
        iterator = self._guarded(lambda: self.source.iter_batches(batch_size))
        for batch in iterator:
            yield self._postprocess(batch, check_rows=False)

    @staticmethod
    def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with snake_case, unique column names."""
        names = [normalize_column_name(c) for c in frame.columns]
        dupes = sorted(n for n, k in Counter(names).items() if k > 1)
        if dupes:
            raise IngestionError(f"Column names collide after normalisation: {dupes}")
        out = frame.copy()
        out.columns = names
        return out

    # -- internals -------------------------------------------------------- #
    def _guarded(self, func: Any) -> Any:
        """Translate low-level exceptions into the ingestion error taxonomy."""
        try:
            return func()
        except IngestionError:
            raise
        except (FileNotFoundError, PermissionError, IsADirectoryError, NotADirectoryError) as exc:
            raise IngestionError(f"{self.source.description}: {exc}") from exc
        except (OSError, TimeoutError, sqlite3.OperationalError) as exc:
            raise TransientIngestionError(f"{self.source.description}: {exc}") from exc
        except ImportError as exc:
            raise IngestionError(f"Optional dependency missing for {self.source.description}: {exc}") from exc
        except (ValueError, KeyError, pd.errors.ParserError) as exc:  # includes EmptyDataError
            raise IngestionError(f"Could not parse {self.source.description}: {exc}") from exc

    def _postprocess(self, frame: Any, *, check_rows: bool) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame):
            raise IngestionError(f"Source returned {type(frame).__name__}, expected DataFrame")
        if self.normalize_columns:
            frame = self.normalize_frame(frame)
        elif frame.columns.duplicated().any():
            raise IngestionError("Duplicate column names in source data")
        missing = [c for c in self.required_columns if c not in frame.columns]
        if missing:
            raise IngestionError(f"Missing required columns: {missing}")
        if check_rows and len(frame) < self.min_rows:
            raise IngestionError(f"Expected at least {self.min_rows} row(s), got {len(frame)}")
        return frame.reset_index(drop=True)


def _as_ingestor(source: Union[DataIngestor, DataSource, str, Path]) -> DataIngestor:
    if isinstance(source, DataIngestor):
        return source
    if isinstance(source, DataSource):
        return DataIngestor(source)
    return DataIngestor.from_path(source)


class IngestionStep(PipelineStep):
    """Pipeline step that loads data into ``context.data``.

    Only :class:`TransientIngestionError` is retried (3 attempts, exponential
    backoff by default); a missing file or malformed content fails fast.
    """

    def __init__(
        self,
        source: Union[DataIngestor, DataSource, str, Path],
        *,
        name: str = "ingestion",
        retry: Optional[RetryPolicy] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name,
            retry=retry
            or RetryPolicy(max_attempts=3, backoff_s=0.5, retry_on=(TransientIngestionError,)),
            **kwargs,
        )
        self.ingestor = _as_ingestor(source)

    def run(self, context: PipelineContext) -> PipelineContext:
        frame = self.ingestor.ingest()
        context.data = frame
        context.metadata["ingestion"] = {
            "source": self.ingestor.source.description,
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "normalize_columns": self.ingestor.normalize_columns,
            "ingested_at": utc_now_iso(),
        }
        return context
