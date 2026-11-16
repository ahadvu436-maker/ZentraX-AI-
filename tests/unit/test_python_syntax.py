import ast
import pytest

from fixtures.project_scan import iter_python_files


pytestmark = pytest.mark.unit


def test_all_project_python_files_parse(project_root):
    files = list(iter_python_files(project_root))
    assert files, f"No Python files found under {project_root}"
    failures = []
    for item in files:
        try:
            ast.parse(item.source, filename=item.relative_path)
        except SyntaxError as exc:
            failures.append(f"{item.relative_path}:{exc.lineno}: {exc.msg}")
    assert not failures, "Python syntax errors:\n" + "\n".join(failures)


def test_python_files_are_not_empty(project_root):
    files = list(iter_python_files(project_root))
    empty = [item.relative_path for item in files if not item.source.strip()]
    assert not empty, "Empty Python files found: " + ", ".join(empty)
