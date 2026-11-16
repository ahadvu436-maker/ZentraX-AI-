"""
Shared pytest configuration for ZentraX AI.

Run from the repository root:
    python -m pytest 19_tests -v

Optional environment variables:
    ZENTRAX_PROJECT_ROOT       Repository root (defaults to parent of this folder)
    ZENTRAX_IMPORT_TARGETS     Comma-separated Python import paths to verify
    ZENTRAX_APP_FACTORY        Import path to a zero-argument app factory, e.g. backend.main:create_app
    ZENTRAX_HEALTH_URL         Optional deployed health URL for system checks
    ZENTRAX_PERF_TARGET        Import path to a zero-argument callable for a basic timing smoke test
    ZENTRAX_ALLOWED_SECURITY_PATTERNS
                               Comma-separated relative-path substrings exempted from static checks
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

TEST_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = TEST_DIR.parent
PROJECT_ROOT = Path(os.getenv("ZENTRAX_PROJECT_ROOT", str(DEFAULT_PROJECT_ROOT))).resolve()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))


def import_from_path(import_path: str) -> Any:
    """Import 'package.module' or 'package.module:attribute' safely for a test."""
    module_path, separator, attribute = import_path.partition(":")
    if not module_path.strip():
        raise ValueError(f"Invalid import path: {import_path!r}")
    module = importlib.import_module(module_path.strip())
    if separator:
        try:
            return getattr(module, attribute.strip())
        except AttributeError as exc:
            raise ImportError(
                f"{attribute!r} was not found in module {module_path!r}"
            ) from exc
    return module


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "unit: fast isolated tests")
    config.addinivalue_line("markers", "integration: tests across module boundaries")
    config.addinivalue_line("markers", "system: end-to-end or deployed-system tests")
    config.addinivalue_line("markers", "performance: timing/load smoke tests")
    config.addinivalue_line("markers", "security: security-focused checks")
    config.addinivalue_line("markers", "regression: tests guarding previously fixed defects")


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def source_root(project_root: Path) -> Path:
    """Best-effort source directory; override ZENTRAX_SOURCE_ROOT if needed."""
    configured = os.getenv("ZENTRAX_SOURCE_ROOT")
    if configured:
        path = Path(configured)
        return (project_root / path).resolve() if not path.is_absolute() else path.resolve()
    for candidate in (project_root / "backend", project_root / "src", project_root):
        if candidate.exists():
            return candidate.resolve()
    return project_root


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def import_targets() -> list[str]:
    raw = os.getenv("ZENTRAX_IMPORT_TARGETS", "")
    return [item.strip() for item in raw.split(",") if item.strip()]


@pytest.fixture
def import_target_loader() -> Callable[[str], Any]:
    return import_from_path


@pytest.fixture
def app_instance() -> Any:
    """Create an app only when ZENTRAX_APP_FACTORY is explicitly configured."""
    target = os.getenv("ZENTRAX_APP_FACTORY", "").strip()
    if not target:
        pytest.skip("Set ZENTRAX_APP_FACTORY=package.module:create_app to enable app tests")
    factory = import_from_path(target)
    if not callable(factory):
        raise TypeError(f"Configured app factory {target!r} is not callable")
    return factory()


@pytest.fixture
def allowed_security_patterns() -> tuple[str, ...]:
    raw = os.getenv("ZENTRAX_ALLOWED_SECURITY_PATTERNS", "")
    return tuple(part.strip() for part in raw.split(",") if part.strip())


@pytest.fixture
def perf_callable() -> Callable[[], Any]:
    target = os.getenv("ZENTRAX_PERF_TARGET", "").strip()
    if not target:
        pytest.skip("Set ZENTRAX_PERF_TARGET=package.module:function to enable perf smoke test")
    func = import_from_path(target)
    if not callable(func):
        raise TypeError(f"Configured performance target {target!r} is not callable")
    return func
