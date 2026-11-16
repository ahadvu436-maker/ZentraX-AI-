"""Static project scanning helpers used by tests."""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class PythonFile:
    path: Path
    relative_path: str
    source: str


def iter_python_files(root: Path) -> Iterator[PythonFile]:
    excluded = {".git", ".venv", "venv", "__pycache__", "node_modules", ".pytest_cache"}
    for path in root.rglob("*.py"):
        if any(part in excluded for part in path.parts):
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            relative = path.as_posix()
        yield PythonFile(path=path, relative_path=relative, source=source)


def parse_python(source: str, filename: str = "<unknown>") -> ast.Module:
    return ast.parse(source, filename=filename)


def package_directories(root: Path) -> list[Path]:
    """Directories containing Python files; namespace packages are allowed."""
    result = []
    for directory in root.rglob("*"):
        if not directory.is_dir():
            continue
        if any(part in {".git", ".venv", "venv", "__pycache__", "node_modules"} for part in directory.parts):
            continue
        if any(directory.glob("*.py")):
            result.append(directory)
    return result
