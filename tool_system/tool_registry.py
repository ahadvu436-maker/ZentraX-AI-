"""Thread-safe registry of tool instances, with automatic registration from discovery."""
from __future__ import annotations

import difflib
import inspect
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Type, Union

from .tool_core import (
    BaseTool, ToolCategory, ToolNotFoundError, ToolRegistrationError, ToolSpec,
)
from .tool_discovery import DiscoveryReport, ToolDiscovery

logger = logging.getLogger(__name__)

__all__ = ["ToolRegistry", "RegistrationReport"]


@dataclass
class RegistrationReport:
    """Result of registering discovered tools."""

    registered: List[str] = field(default_factory=list)
    skipped: Dict[str, str] = field(default_factory=dict)   # tool name -> reason
    errors: Dict[str, str] = field(default_factory=dict)    # module/tool -> error text

    def summary(self) -> str:
        return (f"{len(self.registered)} registered, {len(self.skipped)} skipped, "
                f"{len(self.errors)} errors")


class ToolRegistry:
    """Holds the live set of tools, keyed by ``spec.name``."""

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}
        self._lock = threading.RLock()

    # -- registration ------------------------------------------------------- #
    def register(self, tool: BaseTool, *, replace: bool = False) -> BaseTool:
        """Register a tool instance (initialising it). Raises on duplicates unless ``replace``."""
        if not isinstance(tool, BaseTool):
            raise ToolRegistrationError(f"{tool!r} is not a BaseTool instance")
        spec = getattr(tool, "spec", None)
        if not isinstance(spec, ToolSpec):
            raise ToolRegistrationError(f"{type(tool).__name__} has no valid ToolSpec")
        with self._lock:
            existing = self._tools.get(spec.name)
            if existing is not None and not replace:
                raise ToolRegistrationError(f"tool '{spec.name}' is already registered")
            try:
                tool.initialize()
            except Exception as exc:  # noqa: BLE001
                raise ToolRegistrationError(f"tool '{spec.name}' failed to initialise: {exc}") from exc
            self._tools[spec.name] = tool
        if existing is not None and existing is not tool:
            self._safe_close(existing)
        logger.debug("registered tool %s", spec.name)
        return tool

    def register_class(self, cls: Type[BaseTool], config: Optional[Mapping[str, Any]] = None,
                       *, replace: bool = False) -> BaseTool:
        """Instantiate ``cls(config)`` and register it."""
        if not (inspect.isclass(cls) and issubclass(cls, BaseTool)) or inspect.isabstract(cls):
            raise ToolRegistrationError(f"{cls!r} is not a concrete BaseTool subclass")
        try:
            tool = cls(config)
        except Exception as exc:  # noqa: BLE001
            raise ToolRegistrationError(f"could not instantiate {cls.__name__}: {exc}") from exc
        return self.register(tool, replace=replace)

    def register_discovered(
        self, report: DiscoveryReport, *, configs: Optional[Mapping[str, Mapping[str, Any]]] = None,
        enabled: Optional[Iterable[str]] = None, disabled: Iterable[str] = (),
        replace: bool = False,
    ) -> RegistrationReport:
        """Register every tool class found by discovery.

        ``configs`` maps tool name -> config dict; ``enabled`` (if given) is an allow-list and
        ``disabled`` a deny-list.  Unavailable tools and name clashes are skipped, never fatal.
        """
        out = RegistrationReport(errors=dict(report.errors))
        allow = set(enabled) if enabled is not None else None
        deny = set(disabled)
        for found in report.tools:
            cls = found.cls
            name = cls.spec.name
            if name in deny or (allow is not None and name not in allow):
                out.skipped[name] = "disabled by configuration"
                continue
            try:
                tool = cls(dict((configs or {}).get(name, {})))
            except Exception as exc:  # noqa: BLE001
                out.errors[f"{found.module}.{cls.__name__}"] = f"instantiation failed: {exc}"
                continue
            if not tool.is_available():
                out.skipped[name] = tool.unavailable_reason() or "unavailable"
                continue
            if name in self and not replace:
                out.skipped[name] = "name already registered"
                continue
            try:
                self.register(tool, replace=replace)
                out.registered.append(name)
            except ToolRegistrationError as exc:
                out.errors[f"{found.module}.{cls.__name__}"] = str(exc)
        logger.info("tool registration: %s", out.summary())
        return out

    def discover_and_register(self, discovery: Optional[ToolDiscovery] = None, **kwargs: Any) -> RegistrationReport:
        """Convenience: run discovery (default: the built-in ``tools`` package) and register."""
        return self.register_discovered((discovery or ToolDiscovery()).discover(), **kwargs)

    def unregister(self, name: str, *, close: bool = True) -> None:
        with self._lock:
            tool = self._tools.pop(name, None)
        if tool is None:
            raise ToolNotFoundError(self._not_found_message(name))
        if close:
            self._safe_close(tool)

    def clear(self) -> None:
        with self._lock:
            tools = list(self._tools.values())
            self._tools.clear()
        for tool in tools:
            self._safe_close(tool)

    # -- lookup ------------------------------------------------------------- #
    def get(self, name: str) -> BaseTool:
        with self._lock:
            tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(self._not_found_message(name))
        return tool

    def has(self, name: str) -> bool:
        with self._lock:
            return name in self._tools

    def names(self) -> List[str]:
        with self._lock:
            return sorted(self._tools)

    def list(self, *, category: Optional[ToolCategory] = None, tag: Optional[str] = None,
             only_available: bool = True) -> List[BaseTool]:
        with self._lock:
            tools = [self._tools[n] for n in sorted(self._tools)]
        return [t for t in tools
                if (category is None or t.spec.category == category)
                and (tag is None or tag in t.spec.tags)
                and (not only_available or t.is_available())]

    def specs(self, **filters: Any) -> List[ToolSpec]:
        return [t.spec for t in self.list(**filters)]

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self.has(name)

    def __len__(self) -> int:
        with self._lock:
            return len(self._tools)

    def __iter__(self) -> Iterator[BaseTool]:
        return iter(self.list(only_available=False))

    # -- helpers ------------------------------------------------------------ #
    def _not_found_message(self, name: str) -> str:
        close = difflib.get_close_matches(name, self.names(), n=3)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        return f"unknown tool '{name}'.{hint}"

    @staticmethod
    def _safe_close(tool: BaseTool) -> None:
        try:
            tool.close()
        except Exception:  # noqa: BLE001
            logger.exception("error closing tool %s", getattr(tool, "name", tool))
