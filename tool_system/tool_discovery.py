"""Dynamic discovery of tool classes.

Scans the built-in ``tools`` sub-package (and, optionally, plugin directories), imports
each module defensively and collects every concrete :class:`BaseTool` subclass defined
there.  A broken module never stops discovery; its error is reported instead.

SECURITY: discovery *imports* (i.e. executes) the modules it scans.  Only point it at
code you trust; plugin directories are therefore opt-in.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import logging
import pkgutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional, Sequence, Type, Union

from .tool_core import BaseTool, ToolDiscoveryError, ToolSpec

logger = logging.getLogger(__name__)

__all__ = ["ToolDiscovery", "DiscoveredTool", "DiscoveryReport"]


@dataclass(frozen=True)
class DiscoveredTool:
    """A tool class found during discovery."""

    cls: Type[BaseTool]
    module: str
    origin: str  # "package" or "path"


@dataclass
class DiscoveryReport:
    """Everything one discovery pass found."""

    tools: List[DiscoveredTool] = field(default_factory=list)
    errors: Dict[str, str] = field(default_factory=dict)  # module -> error
    modules_scanned: List[str] = field(default_factory=list)

    @property
    def names(self) -> List[str]:
        return sorted(t.cls.spec.name for t in self.tools)


class ToolDiscovery:
    """Finds :class:`BaseTool` subclasses in a package and/or plugin directories.

    Args:
        package: Dotted package to scan (default: the ``tools`` sub-package of this package).
        plugin_paths: Extra directories of ``*.py`` tool modules (opt-in, trusted code only).
    """

    def __init__(self, package: Optional[str] = None,
                 plugin_paths: Sequence[Union[str, Path]] = ()) -> None:
        base = __package__ or "tool_system"
        self.package = package if package is not None else f"{base}.tools"
        self.plugin_paths = [Path(p) for p in plugin_paths]

    def discover(self, *, reload: bool = False) -> DiscoveryReport:
        """Scan everything and return the report. ``reload=True`` re-imports modules."""
        report = DiscoveryReport()
        seen: set = set()
        if self.package:
            self._scan_package(report, seen, reload)
        for path in self.plugin_paths:
            self._scan_path(path, report, seen)
        return report

    # -- package scanning --------------------------------------------------- #
    def _scan_package(self, report: DiscoveryReport, seen: set, reload: bool) -> None:
        try:
            pkg = importlib.import_module(self.package)
        except Exception as exc:  # noqa: BLE001
            raise ToolDiscoveryError(f"cannot import tool package '{self.package}': {exc}") from exc
        search = getattr(pkg, "__path__", None)
        if search is None:
            raise ToolDiscoveryError(f"'{self.package}' is not a package")
        for info in sorted(pkgutil.iter_modules(search), key=lambda m: m.name):
            if info.name.startswith("_"):
                continue
            full = f"{self.package}.{info.name}"
            report.modules_scanned.append(full)
            try:
                module = importlib.import_module(full)
                if reload:
                    module = importlib.reload(module)
            except Exception as exc:  # noqa: BLE001 - includes SyntaxError / ImportError
                logger.warning("tool module %s failed to import: %s", full, exc)
                report.errors[full] = f"{type(exc).__name__}: {exc}"
                continue
            self._collect(module, "package", report, seen)

    # -- plugin directory scanning ------------------------------------------ #
    def _scan_path(self, directory: Path, report: DiscoveryReport, seen: set) -> None:
        if not directory.is_dir():
            report.errors[str(directory)] = "plugin path is not a directory"
            return
        for file in sorted(directory.glob("*.py")):
            if file.name.startswith("_"):
                continue
            digest = hashlib.sha1(str(file.resolve()).encode()).hexdigest()[:8]
            mod_name = f"_tool_plugin_{digest}_{file.stem}"
            report.modules_scanned.append(str(file))
            try:
                spec = importlib.util.spec_from_file_location(mod_name, file)
                if spec is None or spec.loader is None:
                    raise ImportError("no loader for file")
                module = importlib.util.module_from_spec(spec)
                sys.modules[mod_name] = module
                try:
                    spec.loader.exec_module(module)
                except BaseException:
                    sys.modules.pop(mod_name, None)
                    raise
            except Exception as exc:  # noqa: BLE001
                logger.warning("plugin %s failed to import: %s", file, exc)
                report.errors[str(file)] = f"{type(exc).__name__}: {exc}"
                continue
            self._collect(module, "path", report, seen)

    # -- class collection --------------------------------------------------- #
    @staticmethod
    def _collect(module: ModuleType, origin: str, report: DiscoveryReport, seen: set) -> None:
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if (cls is BaseTool or not issubclass(cls, BaseTool)
                    or cls.__module__ != module.__name__ or inspect.isabstract(cls)):
                continue
            spec = getattr(cls, "spec", None)
            if not isinstance(spec, ToolSpec):
                report.errors[f"{module.__name__}.{cls.__name__}"] = "missing class-level ToolSpec"
                continue
            if cls in seen:
                continue
            seen.add(cls)
            report.tools.append(DiscoveredTool(cls, module.__name__, origin))
