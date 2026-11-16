"""Content-addressed dataset versioning (identical data => identical version id)."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .core import PipelineContext, PipelineError, Record, Stage


@dataclass(frozen=True)
class DatasetVersion:
    name: str
    version: str
    record_count: int
    created_at: str
    path: str


class DatasetVersionStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    @staticmethod
    def _digest(lines: list[str]) -> str:
        h = hashlib.sha256()
        for line in lines:
            h.update(line.encode())
            h.update(b"\n")
        return h.hexdigest()[:16]

    def save(self, name: str, records: list[Record], metadata: dict[str, Any] | None = None) -> DatasetVersion:
        lines = [json.dumps(r, sort_keys=True, default=str) for r in records]
        version = self._digest(lines)
        target = self.root / name / version
        manifest_path = target / "manifest.json"
        if manifest_path.exists():  # idempotent
            return DatasetVersion(**json.loads(manifest_path.read_text())["version_info"])
        target.mkdir(parents=True, exist_ok=True)
        info = DatasetVersion(name, version, len(records), datetime.now(timezone.utc).isoformat(), str(target))
        self._atomic_write(target / "data.jsonl", "\n".join(lines) + ("\n" if lines else ""))
        # manifest last: its presence marks a complete version
        self._atomic_write(
            manifest_path, json.dumps({"version_info": asdict(info), "metadata": metadata or {}}, indent=2)
        )
        return info

    def load(self, name: str, version: str) -> list[Record]:
        path = self.root / name / version / "data.jsonl"
        if not path.exists():
            raise PipelineError(f"dataset {name}@{version} not found")
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    def latest(self, name: str) -> DatasetVersion | None:
        manifests = sorted(
            (self.root / name).glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        return DatasetVersion(**json.loads(manifests[0].read_text())["version_info"]) if manifests else None

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        fd, tmp = tempfile.mkstemp(dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(content)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


class VersioningStage(Stage[list[Record], list[Record]]):
    name = "versioning"

    def __init__(self, store: DatasetVersionStore, dataset_name: str) -> None:
        super().__init__()
        self.store, self.dataset_name = store, dataset_name

    def process(self, data: list[Record], ctx: PipelineContext) -> list[Record]:
        info = self.store.save(self.dataset_name, data, {"run_id": ctx.run_id})
        ctx.metadata["dataset_version"] = info
        return data
