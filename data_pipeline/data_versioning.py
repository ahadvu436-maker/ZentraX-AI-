"""Content-addressed dataset versioning.

:class:`DataVersionStore` is a small, dependency-free registry that keeps
immutable, numbered snapshots of DataFrames:

* every ``commit`` is identified by ``<name>@v<N>`` and a SHA-256 *content
  hash* (identical data is not stored twice),
* versions form a lineage (``parent``), e.g. raw -> clean -> features,
* ``load`` re-hashes the data and refuses to return a snapshot that was
  modified on disk,
* tags (``"production"``) can point at a version; moving a tag is atomic.

Storage: Parquet when ``pyarrow``/``fastparquet`` is installed, otherwise
deterministic ``csv.gz`` plus recorded dtypes. The CSV fallback cannot tell an
empty string from a missing value; the store verifies the round trip at commit
time and refuses data it cannot reproduce exactly.

Concurrency: registry updates take a lock file, so several processes can
commit safely on a local filesystem.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import importlib.util
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Union

import pandas as pd

from .pipeline_core import (
    PipelineContext,
    PipelineError,
    PipelineStep,
    atomic_write_json,
    utc_now_iso,
)

__all__ = [
    "DataVersioningError",
    "DataVersion",
    "DataVersionStore",
    "compute_content_hash",
    "VersioningStep",
]

logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_TAG_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,63}$")
_VERSION_ID_RE = re.compile(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9_.-]*)@v(?P<version>\d+)$")
_VERSION_REF_RE = re.compile(r"^v?(\d+)$")


class DataVersioningError(PipelineError):
    """Versioning operation failed or an integrity check did not pass."""


def compute_content_hash(df: pd.DataFrame, chunk_rows: int = 50_000) -> str:
    """Deterministic SHA-256 of a frame's column names and cell values.

    The hash is computed over a canonical CSV rendering (index excluded), so it
    does not depend on the in-memory dtype flavour (``object`` vs ``str``) and
    survives a storage round trip. Memory use is bounded by ``chunk_rows``.
    """
    digest = hashlib.sha256()
    digest.update(json.dumps([str(c) for c in df.columns]).encode("utf-8"))
    digest.update(f"|{len(df)}|".encode("ascii"))
    for start in range(0, len(df), chunk_rows):
        chunk = df.iloc[start : start + chunk_rows]
        digest.update(chunk.to_csv(index=False, header=False, lineterminator="\n").encode("utf-8"))
    return digest.hexdigest()


@dataclass
class DataVersion:
    """Metadata of one immutable snapshot."""

    name: str
    version: int
    content_hash: str
    created_at: str
    rows: int
    columns: List[str]
    dtypes: Dict[str, str]
    storage_path: str
    storage_format: str
    parent: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def version_id(self) -> str:
        return f"{self.name}@v{self.version}"

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DataVersion":
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in payload.items() if k in names})


def _has_parquet_engine() -> bool:
    return any(importlib.util.find_spec(m) is not None for m in ("pyarrow", "fastparquet"))


class DataVersionStore:
    """Registry of immutable dataset snapshots under ``root``.

    Args:
        root: Directory holding ``registry.json`` and ``data/``.
        storage_format: ``"auto"`` (Parquet if available, else ``csv.gz``),
            ``"parquet"`` or ``"csv.gz"``.
        lock_timeout_s: How long to wait for the registry lock.
    """

    REGISTRY_FILE = "registry.json"
    LOCK_FILE = ".registry.lock"
    _STALE_LOCK_S = 300.0

    def __init__(
        self,
        root: Union[str, Path],
        *,
        storage_format: str = "auto",
        lock_timeout_s: float = 30.0,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock_timeout_s = lock_timeout_s
        if storage_format == "auto":
            storage_format = "parquet" if _has_parquet_engine() else "csv.gz"
        if storage_format not in ("parquet", "csv.gz"):
            raise DataVersioningError("storage_format must be 'auto', 'parquet' or 'csv.gz'")
        if storage_format == "parquet" and not _has_parquet_engine():
            raise DataVersioningError("Parquet storage needs pyarrow or fastparquet")
        self.storage_format = storage_format

    # -- registry I/O ------------------------------------------------------ #
    @property
    def _registry_path(self) -> Path:
        return self.root / self.REGISTRY_FILE

    def _read_registry(self) -> Dict[str, Any]:
        if not self._registry_path.is_file():
            return {"format_version": 1, "datasets": {}}
        try:
            return json.loads(self._registry_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise DataVersioningError(f"Corrupt registry {self._registry_path}: {exc}") from exc

    def _write_registry(self, registry: Mapping[str, Any]) -> None:
        atomic_write_json(self._registry_path, registry)

    @contextlib.contextmanager
    def _lock(self) -> Iterator[None]:
        lock_path = self.root / self.LOCK_FILE
        deadline = time.monotonic() + self.lock_timeout_s
        while True:
            try:
                os.close(os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                break
            except FileExistsError:
                with contextlib.suppress(OSError):
                    if time.time() - lock_path.stat().st_mtime > self._STALE_LOCK_S:
                        lock_path.unlink()
                        continue
                if time.monotonic() > deadline:
                    raise DataVersioningError(
                        f"Timed out waiting for {lock_path}; remove it if no process is committing"
                    )
                time.sleep(0.05)
        try:
            yield
        finally:
            with contextlib.suppress(FileNotFoundError):
                lock_path.unlink()

    def _safe_path(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise DataVersioningError(f"Path escapes the store root: {relative}")
        return path

    @staticmethod
    def _check_name(name: str) -> None:
        if not isinstance(name, str) or not _NAME_RE.match(name):
            raise DataVersioningError(
                f"Invalid dataset name {name!r}: use letters, digits, '_', '.', '-'"
            )

    @staticmethod
    def _check_tags(tags: Sequence[str]) -> None:
        for tag in tags:
            if not _TAG_RE.match(tag) or tag == "latest" or _VERSION_REF_RE.match(tag):
                raise DataVersioningError(f"Invalid tag {tag!r}")

    # -- reference resolution ---------------------------------------------- #
    def _entries(self, registry: Mapping[str, Any], name: str) -> List[Dict[str, Any]]:
        self._check_name(name)
        entries = registry["datasets"].get(name)
        if not entries:
            raise DataVersioningError(f"Unknown dataset '{name}'")
        return entries

    def _resolve_entry(self, entries: List[Dict[str, Any]], ref: Union[int, str]) -> Dict[str, Any]:
        if isinstance(ref, int) and not isinstance(ref, bool):
            number: Optional[int] = ref
        elif ref == "latest":
            return entries[-1]
        elif isinstance(ref, str) and _VERSION_REF_RE.match(ref):
            number = int(_VERSION_REF_RE.match(ref).group(1))  # type: ignore[union-attr]
        else:
            tagged = [e for e in entries if ref in e.get("tags", [])]
            if not tagged:
                raise DataVersioningError(f"No version with tag or reference {ref!r}")
            return tagged[-1]
        for entry in entries:
            if entry["version"] == number:
                return entry
        raise DataVersioningError(f"Version {number} does not exist")

    def get(self, name: str, ref: Union[int, str] = "latest") -> DataVersion:
        """Metadata of a version. ``ref``: ``"latest"``, ``3``, ``"v3"`` or a tag."""
        registry = self._read_registry()
        return DataVersion.from_dict(self._resolve_entry(self._entries(registry, name), ref))

    def get_by_id(self, version_id: str) -> DataVersion:
        """Metadata of a version given ``"name@vN"``."""
        match = _VERSION_ID_RE.match(version_id or "")
        if not match:
            raise DataVersioningError(f"Malformed version id {version_id!r}")
        return self.get(match["name"], int(match["version"]))

    def list_datasets(self) -> List[str]:
        return sorted(self._read_registry()["datasets"])

    def list_versions(self, name: str) -> List[DataVersion]:
        return [DataVersion.from_dict(e) for e in self._entries(self._read_registry(), name)]

    # -- commit ------------------------------------------------------------ #
    def commit(
        self,
        name: str,
        df: pd.DataFrame,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
        tags: Sequence[str] = (),
        parent: Optional[str] = None,
        deduplicate: bool = True,
    ) -> DataVersion:
        """Store ``df`` as a new version of ``name`` and return its metadata.

        Args:
            metadata: Free-form JSON-able facts (stage, config hash, ...).
            tags: Labels to attach; a tag already used by another version of
                this dataset is *moved* to the new one.
            parent: ``version_id`` this data was derived from.
            deduplicate: Return the latest version unchanged when its content
                hash equals the new data's.
        """
        self._check_name(name)
        self._check_tags(tags)
        if not isinstance(df, pd.DataFrame):
            raise DataVersioningError(f"Expected DataFrame, got {type(df).__name__}")
        frame = df.reset_index(drop=True)
        content_hash = compute_content_hash(frame)

        with self._lock():
            registry = self._read_registry()
            entries: List[Dict[str, Any]] = registry["datasets"].setdefault(name, [])
            if parent is not None:
                self.get_by_id(parent)

            if deduplicate and entries and entries[-1]["content_hash"] == content_hash:
                latest = entries[-1]
                if tags:
                    self._apply_tags(entries, latest, tags)
                    self._write_registry(registry)
                logger.info("Data unchanged; reusing %s@v%d", name, latest["version"])
                return DataVersion.from_dict(latest)

            number = entries[-1]["version"] + 1 if entries else 1
            relative = f"data/{name}/v{number}.{self.storage_format}"
            path = self._safe_path(relative)
            self._write_frame(frame, path)
            version = DataVersion(
                name=name, version=number, content_hash=content_hash,
                created_at=utc_now_iso(), rows=len(frame),
                columns=[str(c) for c in frame.columns],
                dtypes={str(c): str(t) for c, t in frame.dtypes.items()},
                storage_path=relative, storage_format=self.storage_format,
                parent=parent, tags=[], metadata=dict(metadata or {}),
            )
            try:
                if self.storage_format == "csv.gz":
                    self._verify_roundtrip(version, path)
            except DataVersioningError:
                path.unlink(missing_ok=True)
                raise
            entry = version.to_dict()
            entries.append(entry)
            if tags:
                self._apply_tags(entries, entry, tags)
            self._write_registry(registry)
        logger.info("Committed %s (%d rows, hash %s...)", version.version_id, version.rows, content_hash[:12])
        return DataVersion.from_dict(entry)

    @staticmethod
    def _apply_tags(entries: List[Dict[str, Any]], target: Dict[str, Any], tags: Sequence[str]) -> None:
        for entry in entries:
            entry["tags"] = [t for t in entry.get("tags", []) if t not in tags]
        target["tags"] = sorted({*target.get("tags", []), *tags})

    def tag(self, name: str, ref: Union[int, str], tag: str) -> DataVersion:
        """Attach ``tag`` to a version (moving it from any other version)."""
        self._check_tags([tag])
        with self._lock():
            registry = self._read_registry()
            entries = self._entries(registry, name)
            target = self._resolve_entry(entries, ref)
            self._apply_tags(entries, target, [tag])
            self._write_registry(registry)
        return DataVersion.from_dict(target)

    # -- load / verify ------------------------------------------------------ #
    def load(self, name: str, ref: Union[int, str] = "latest", *, verify: bool = True) -> pd.DataFrame:
        """Load a snapshot; with ``verify`` the content hash must still match."""
        version = self.get(name, ref)
        frame = self._read_frame(version)
        if verify and compute_content_hash(frame) != version.content_hash:
            raise DataVersioningError(
                f"Integrity check failed for {version.version_id}: stored data was modified"
            )
        return frame

    def verify(self, name: str, ref: Union[int, str] = "latest") -> bool:
        """``True`` if the stored snapshot still matches its recorded hash."""
        try:
            self.load(name, ref, verify=True)
        except (DataVersioningError, OSError, ValueError):
            return False
        return True

    def lineage(self, name: str, ref: Union[int, str] = "latest") -> List[DataVersion]:
        """Ancestors-first chain ending at the requested version."""
        chain: List[DataVersion] = []
        seen = set()
        current: Optional[DataVersion] = self.get(name, ref)
        while current is not None and current.version_id not in seen:
            seen.add(current.version_id)
            chain.append(current)
            current = self.get_by_id(current.parent) if current.parent else None
        return chain[::-1]

    def diff(self, name: str, ref_a: Union[int, str], ref_b: Union[int, str]) -> Dict[str, Any]:
        """Compare two versions using metadata only (no data is loaded)."""
        a, b = self.get(name, ref_a), self.get(name, ref_b)
        common = set(a.columns) & set(b.columns)
        return {
            "from": a.version_id,
            "to": b.version_id,
            "identical": a.content_hash == b.content_hash,
            "rows_delta": b.rows - a.rows,
            "columns_added": [c for c in b.columns if c not in set(a.columns)],
            "columns_removed": [c for c in a.columns if c not in set(b.columns)],
            "dtype_changes": {
                c: [a.dtypes[c], b.dtypes[c]] for c in sorted(common) if a.dtypes[c] != b.dtypes[c]
            },
        }

    # -- storage ------------------------------------------------------------ #
    def _write_frame(self, frame: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        try:
            if self.storage_format == "parquet":
                frame.to_parquet(tmp, index=False)
            else:
                frame.to_csv(tmp, index=False, compression={"method": "gzip", "mtime": 0})
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    def _read_frame(self, version: DataVersion) -> pd.DataFrame:
        path = self._safe_path(version.storage_path)
        if not path.is_file():
            raise DataVersioningError(f"Data file missing for {version.version_id}: {path}")
        if version.storage_format == "parquet":
            return pd.read_parquet(path)

        datetime_cols = [c for c, t in version.dtypes.items() if t.startswith("datetime64")]
        dtype_map = {c: t for c, t in version.dtypes.items() if c not in datetime_cols}
        frame = pd.read_csv(
            path, compression="gzip", dtype=dtype_map,
            keep_default_na=False, na_values=[""], float_precision="round_trip",
        )
        for col in datetime_cols:
            dtype = version.dtypes[col]
            parsed = pd.to_datetime(frame[col], utc="," in dtype, errors="coerce")
            try:
                parsed = parsed.astype(dtype)
            except (TypeError, ValueError):
                pass
            frame[col] = parsed
        return frame[version.columns]

    def _verify_roundtrip(self, version: DataVersion, path: Path) -> None:
        try:
            reloaded = self._read_frame(version)
        except (ValueError, TypeError) as exc:
            raise DataVersioningError(
                f"Data cannot be stored losslessly as csv.gz ({exc}); install pyarrow for Parquet"
            ) from exc
        if compute_content_hash(reloaded) != version.content_hash:
            raise DataVersioningError(
                "Data does not survive a csv.gz round trip (e.g. empty strings vs missing "
                "values); install pyarrow to use Parquet"
            )


class VersioningStep(PipelineStep):
    """Commit ``context.data`` to a :class:`DataVersionStore`.

    The id of each commit is recorded in ``context.metadata["versions"]`` and
    the next commit automatically uses it as ``parent`` (``link_to_previous``),
    producing a raw -> clean -> features lineage.
    """

    def __init__(
        self,
        store: DataVersionStore,
        dataset: str,
        *,
        tags: Sequence[str] = (),
        metadata: Optional[Mapping[str, Any]] = None,
        link_to_previous: bool = True,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name or f"version_{dataset}", **kwargs)
        self.store = store
        self.dataset = dataset
        self.tags = tuple(tags)
        self.extra_metadata = dict(metadata or {})
        self.link_to_previous = link_to_previous

    def run(self, context: PipelineContext) -> PipelineContext:
        frame = context.require_data()
        order: List[str] = context.metadata.setdefault("version_order", [])
        parent = order[-1] if (self.link_to_previous and order) else None
        meta = {"run_id": context.run_id, "mode": context.mode, **self.extra_metadata}
        version = self.store.commit(
            self.dataset, frame, metadata=meta, tags=self.tags, parent=parent
        )
        context.metadata.setdefault("versions", {})[self.dataset] = version.version_id
        order.append(version.version_id)
        context.artifacts.setdefault("data_versions", {})[self.dataset] = version
        return context
