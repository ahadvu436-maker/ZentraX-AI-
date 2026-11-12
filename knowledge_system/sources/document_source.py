"""Ingest local files (text, markdown, JSON, CSV, HTML, DOCX, PDF) as documents."""
from __future__ import annotations

import csv
import io
import json
import logging
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Union
from xml.etree import ElementTree as ET

from . import BaseSource, RawDocument, html_to_text, make_doc_id

logger = logging.getLogger(__name__)

__all__ = ["DocumentSource"]

_TEXT_EXT = {".txt", ".md", ".markdown", ".rst", ".log", ".text"}
_DEFAULT_EXT = _TEXT_EXT | {
    ".json", ".jsonl", ".csv", ".tsv", ".html", ".htm", ".docx", ".pdf",
}
_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_MAX_DOCX_XML_BYTES = 50 * 1024 * 1024


def _json_lines(obj: Any, prefix: str = "") -> List[str]:
    """Flatten JSON into ``path: value`` lines."""
    lines: List[str] = []
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            lines.extend(_json_lines(value, path))
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            lines.extend(_json_lines(value, f"{prefix}[{i}]"))
    elif obj is not None:
        lines.append(f"{prefix}: {obj}" if prefix else str(obj))
    return lines


def _json_to_text(obj: Any) -> str:
    if isinstance(obj, list):
        return "\n\n".join("\n".join(_json_lines(item)) for item in obj)
    return "\n".join(_json_lines(obj))


class DocumentSource(BaseSource):
    """Read documents from files, directories, or in-memory strings.

    Args:
        paths: Files and/or directories to ingest.
        recursive: Descend into sub-directories.
        extensions: Allowed suffixes (lower-case, with dot). Defaults to all supported.
        max_file_bytes: Files larger than this are skipped.
        encoding: Encoding for text-like files (undecodable bytes are replaced).
        inline: Optional ``{uri: text}`` mapping of in-memory documents.
    """

    source_type = "document"

    def __init__(
        self,
        paths: Sequence[Union[str, Path]] = (),
        *,
        recursive: bool = True,
        extensions: Optional[Iterable[str]] = None,
        max_file_bytes: int = 20 * 1024 * 1024,
        encoding: str = "utf-8",
        inline: Optional[Mapping[str, str]] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name or "documents")
        self.paths = [Path(p).expanduser() for p in paths]
        self.recursive = recursive
        self.extensions: Set[str] = {e.lower() for e in (extensions or _DEFAULT_EXT)}
        self.max_file_bytes = max_file_bytes
        self.encoding = encoding
        self.inline: Dict[str, str] = dict(inline or {})

    @classmethod
    def from_texts(cls, texts: Mapping[str, str], **kwargs: Any) -> "DocumentSource":
        return cls(inline=texts, **kwargs)

    # ------------------------------------------------------------------ #
    def fetch(self) -> Iterator[RawDocument]:
        for uri, text in self.inline.items():
            if text and text.strip():
                yield RawDocument(
                    id=make_doc_id(uri), uri=uri, content=text,
                    source_type=self.source_type, title=uri,
                    metadata={"inline": True},
                )
        for path in self._iter_files():
            try:
                doc = self._load(path)
            except Exception as exc:  # one bad file must not stop ingestion
                logger.warning("Skipping %s: %s", path, exc)
                continue
            if doc is not None:
                yield doc

    def health_check(self) -> bool:
        return bool(self.inline) or any(p.exists() for p in self.paths)

    # ------------------------------------------------------------------ #
    def _iter_files(self) -> Iterator[Path]:
        for root in self.paths:
            if root.is_file():
                yield root
            elif root.is_dir():
                pattern = "**/*" if self.recursive else "*"
                for path in sorted(root.glob(pattern)):
                    if (
                        path.is_file()
                        and not path.name.startswith(".")
                        and path.suffix.lower() in self.extensions
                    ):
                        yield path
            else:
                logger.warning("Path does not exist: %s", root)

    def _load(self, path: Path) -> Optional[RawDocument]:
        stat = path.stat()
        if stat.st_size > self.max_file_bytes:
            logger.warning("Skipping %s: larger than %d bytes", path, self.max_file_bytes)
            return None
        suffix = path.suffix.lower()
        title = path.stem
        if suffix in (".html", ".htm"):
            parsed = html_to_text(self._read_text(path))
            text, title = parsed.text, parsed.title or title
        else:
            text = self._extract(path, suffix)
        if not text or not text.strip():
            return None
        uri = path.resolve().as_uri()
        return RawDocument(
            id=make_doc_id(uri),
            uri=uri,
            content=text,
            source_type=self.source_type,
            title=title,
            metadata={
                "path": str(path),
                "suffix": suffix,
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(),
            },
        )

    def _read_text(self, path: Path) -> str:
        return path.read_text(encoding=self.encoding, errors="replace")

    def _extract(self, path: Path, suffix: str) -> str:
        if suffix == ".json":
            return _json_to_text(json.loads(self._read_text(path)))
        if suffix == ".jsonl":
            items = [json.loads(ln) for ln in self._read_text(path).splitlines() if ln.strip()]
            return _json_to_text(items)
        if suffix in (".csv", ".tsv"):
            delim = "\t" if suffix == ".tsv" else ","
            reader = csv.DictReader(io.StringIO(self._read_text(path)), delimiter=delim)
            rows = ["\n".join(f"{k}: {v}" for k, v in row.items() if k and v) for row in reader]
            return "\n\n".join(r for r in rows if r)
        if suffix == ".docx":
            return self._read_docx(path)
        if suffix == ".pdf":
            return self._read_pdf(path)
        return self._read_text(path)

    @staticmethod
    def _read_docx(path: Path) -> str:
        with zipfile.ZipFile(path) as zf:
            info = zf.getinfo("word/document.xml")
            if info.file_size > _MAX_DOCX_XML_BYTES:
                raise ValueError("document.xml too large")
            root = ET.fromstring(zf.read(info))
        paragraphs = []
        for para in root.iter(f"{_W_NS}p"):
            text = "".join(t.text or "" for t in para.iter(f"{_W_NS}t"))
            if text.strip():
                paragraphs.append(text)
        return "\n".join(paragraphs)

    @staticmethod
    def _read_pdf(path: Path) -> str:
        try:
            from pypdf import PdfReader  # type: ignore
        except ImportError as exc:
            raise ImportError("PDF support requires `pip install pypdf`") from exc
        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
