"""File operations tool, confined to a root directory.

Security model: every path is resolved (following symlinks) and must stay inside the root;
absolute paths and ``..`` escapes are rejected; sensitive names (``.git``, ``.env`` by default)
are denied; writes are atomic and size-capped.  Reads need ``fs.read``; ``write``/``append``/
``mkdir``/``delete`` need ``fs.write`` (checked per action).

The root is ``config["root"]`` or, if absent, ``context.working_dir``.

Config keys: ``root``, ``max_read_bytes``, ``max_write_bytes``, ``denied_names``.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set

from ..tool_core import (
    BaseTool, ToolCategory, ToolContext, ToolExecutionError, ToolParameter, ToolPermissionError,
    ToolSpec, ToolValidationError,
)

_READ_ACTIONS = {"read", "list", "exists", "stat"}
_WRITE_ACTIONS = {"write", "append", "mkdir", "delete"}
_MAX_LIST = 500


class FileTool(BaseTool):
    """Read, write and manage files inside a sandbox directory."""

    spec = ToolSpec(
        name="file_ops",
        description=("Work with files inside the workspace: read, write, append, list, exists, stat, "
                     "mkdir, delete (files or empty directories). Paths are relative to the workspace."),
        parameters=(
            ToolParameter("action", "string", "Operation to perform.",
                          enum=("read", "write", "append", "list", "exists", "stat", "mkdir", "delete")),
            ToolParameter("path", "string", "Path relative to the workspace root ('.' for the root).",
                          max_length=1024),
            ToolParameter("content", "string", "Text to write/append (write and append only).",
                          required=False, default=None),
            ToolParameter("overwrite", "boolean", "Allow overwriting an existing file on write.",
                          required=False, default=True),
            ToolParameter("max_bytes", "integer", "Maximum bytes to read.", required=False,
                          default=100_000, minimum=1, maximum=5_000_000),
        ),
        category=ToolCategory.FILESYSTEM, tags=("files", "io"),
        required_permissions=frozenset({"fs.read"}), timeout=30.0,
    )

    def required_permissions_for(self, arguments: Mapping[str, Any]) -> Set[str]:
        return {"fs.write"} if arguments.get("action") in _WRITE_ACTIONS else {"fs.read"}

    # -- path safety --------------------------------------------------------- #
    def _root(self, context: ToolContext) -> Path:
        root = self.config.get("root") or context.working_dir
        if not root:
            raise ToolValidationError("no workspace root configured (set config 'root' or working_dir)")
        path = Path(root).expanduser().resolve()
        if not path.is_dir():
            raise ToolExecutionError(f"workspace root '{path}' is not a directory")
        return path

    def _resolve(self, root: Path, rel: str) -> Path:
        if "\x00" in rel:
            raise ToolValidationError("invalid path")
        candidate = Path(rel)
        if candidate.is_absolute():
            raise ToolPermissionError("absolute paths are not allowed; use a path relative to the workspace")
        target = (root / candidate).resolve()
        if target != root and root not in target.parents:
            raise ToolPermissionError("path escapes the workspace")
        denied = set(self.config.get("denied_names", (".git", ".env")))
        if denied & set(target.relative_to(root).parts):
            raise ToolPermissionError("access to this path is denied")
        return target

    # -- actions -------------------------------------------------------------- #
    def run(self, context: ToolContext, action: str, path: str, content: Optional[str] = None,
            overwrite: bool = True, max_bytes: int = 100_000, **_: Any) -> Any:
        root = self._root(context)
        target = self._resolve(root, path)
        context.check_cancelled()
        rel = str(target.relative_to(root)) or "."

        if action == "exists":
            return {"path": rel, "exists": target.exists()}
        if action == "stat":
            return self._stat(target, rel)
        if action == "list":
            return self._list(target, root, rel)
        if action == "read":
            return self._read(target, rel, min(max_bytes, int(self.config.get("max_read_bytes", 5_000_000))))
        if action in ("write", "append"):
            return self._write(target, rel, content, append=(action == "append"), overwrite=overwrite)
        if action == "mkdir":
            target.mkdir(parents=True, exist_ok=True)
            return {"path": rel, "created": True}
        if action == "delete":
            if target == root:
                raise ToolPermissionError("cannot delete the workspace root")
            if not target.exists() and not target.is_symlink():
                raise ToolExecutionError(f"'{rel}' does not exist")
            if target.is_dir():
                target.rmdir()  # only empty directories
            else:
                target.unlink()
            return {"path": rel, "deleted": True}
        raise ToolValidationError(f"unknown action '{action}'")  # pragma: no cover (enum-validated)

    @staticmethod
    def _stat(target: Path, rel: str) -> Dict[str, Any]:
        if not target.exists():
            raise ToolExecutionError(f"'{rel}' does not exist")
        st = target.stat()
        return {"path": rel, "type": "directory" if target.is_dir() else "file",
                "size": st.st_size, "modified": st.st_mtime}

    @staticmethod
    def _list(target: Path, root: Path, rel: str) -> Dict[str, Any]:
        if not target.is_dir():
            raise ToolExecutionError(f"'{rel}' is not a directory")
        entries: List[Dict[str, Any]] = []
        for child in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            try:
                is_dir = child.is_dir()
                entries.append({"name": child.name, "type": "directory" if is_dir else "file",
                                "size": None if is_dir else child.stat().st_size})
            except OSError:
                continue
            if len(entries) >= _MAX_LIST:
                break
        return {"path": rel, "entries": entries, "truncated": len(entries) >= _MAX_LIST}

    @staticmethod
    def _read(target: Path, rel: str, limit: int) -> Dict[str, Any]:
        if not target.is_file():
            raise ToolExecutionError(f"'{rel}' is not a file")
        size = target.stat().st_size
        with open(target, "rb") as fh:
            data = fh.read(limit)
        return {"path": rel, "content": data.decode("utf-8", errors="replace"),
                "size": size, "truncated": size > limit}

    def _write(self, target: Path, rel: str, content: Optional[str], *, append: bool, overwrite: bool) -> Dict[str, Any]:
        if content is None:
            raise ToolValidationError("'content' is required for write/append")
        data = content.encode("utf-8")
        if len(data) > int(self.config.get("max_write_bytes", 5_000_000)):
            raise ToolValidationError("content exceeds the maximum write size")
        if target.is_dir():
            raise ToolExecutionError(f"'{rel}' is a directory")
        if not target.parent.is_dir():
            raise ToolExecutionError(f"parent directory of '{rel}' does not exist (use mkdir first)")
        if append:
            with open(target, "ab") as fh:
                fh.write(data)
        else:
            if target.exists() and not overwrite:
                raise ToolExecutionError(f"'{rel}' already exists and overwrite is false")
            fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".tmp_")
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(data)
                os.replace(tmp, target)  # atomic
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        return {"path": rel, "bytes_written": len(data), "mode": "append" if append else "write"}
