"""High-level facade that wires discovery, registry, executor and function calling together."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

from .function_calling import FunctionCallingAdapter, Provider, ToolCall
from .tool_core import BaseTool, ToolCategory, ToolContext, ToolResult, ToolSpec
from .tool_discovery import ToolDiscovery
from .tool_executor import ToolExecutor
from .tool_registry import RegistrationReport, ToolRegistry

logger = logging.getLogger(__name__)

__all__ = ["ToolManager", "DEFAULT_PERMISSIONS"]

#: Safe-by-default grants: reading and network, but no writes or code execution.
DEFAULT_PERMISSIONS = frozenset({"fs.read", "net", "db.read"})


class ToolManager:
    """One object to load, inspect and run tools.

    Typical use::

        with ToolManager(tool_configs={"file_ops": {"root": "/srv/work"}},
                         default_permissions={"fs.read", "fs.write", "net"}) as tm:
            result = tm.call("file_ops", {"action": "list", "path": "."})
            tools = tm.get_tool_definitions("openai")           # send to the model
            calls, results, msgs = tm.handle_model_response(resp, "openai")

    Args:
        tool_configs: ``{tool_name: config_dict}`` passed to each tool's constructor.
        enabled / disabled: allow-list / deny-list of tool names.
        plugin_paths: extra directories with trusted tool modules.
        default_permissions: permissions granted to contexts the manager creates.
        working_dir: default working directory placed in new contexts.
        auto_load: discover and register tools immediately.
    """

    def __init__(
        self, *, tool_configs: Optional[Mapping[str, Mapping[str, Any]]] = None,
        enabled: Optional[Iterable[str]] = None, disabled: Iterable[str] = (),
        plugin_paths: Sequence[Union[str, Path]] = (),
        default_permissions: Iterable[str] = DEFAULT_PERMISSIONS,
        working_dir: Optional[Union[str, Path]] = None, max_concurrency: int = 8,
        default_timeout: float = 30.0, max_output_chars: int = 50_000,
        registry: Optional[ToolRegistry] = None, discovery: Optional[ToolDiscovery] = None,
        auto_load: bool = True,
    ) -> None:
        self.tool_configs = {k: dict(v) for k, v in (tool_configs or {}).items()}
        self.enabled = set(enabled) if enabled is not None else None
        self.disabled = set(disabled)
        self.default_permissions: Set[str] = set(default_permissions)
        self.working_dir = Path(working_dir) if working_dir else None
        self.registry = registry or ToolRegistry()
        self.discovery = discovery or ToolDiscovery(plugin_paths=plugin_paths)
        self.executor = ToolExecutor(
            self.registry, max_concurrency=max_concurrency, default_timeout=default_timeout,
            max_output_chars=max_output_chars, default_context=self.new_context)
        self.function_calling = FunctionCallingAdapter(self.registry, self.executor)
        self.last_report: Optional[RegistrationReport] = None
        if auto_load:
            self.load_tools()

    # -- lifecycle ---------------------------------------------------------- #
    def __enter__(self) -> "ToolManager":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.shutdown()

    def shutdown(self) -> None:
        """Stop the executor and close every tool (releasing connections etc.)."""
        self.executor.shutdown()
        self.registry.clear()

    # -- loading ------------------------------------------------------------ #
    def load_tools(self, *, replace: bool = False, reload: bool = False) -> RegistrationReport:
        """Discover tool classes and register them according to config/allow/deny lists."""
        report = self.discovery.discover(reload=reload)
        self.last_report = self.registry.register_discovered(
            report, configs=self.tool_configs, enabled=self.enabled, disabled=self.disabled,
            replace=replace)
        for module, error in self.last_report.errors.items():
            logger.warning("tool load problem in %s: %s", module, error)
        return self.last_report

    def reload_tools(self) -> RegistrationReport:
        """Re-scan and replace tools (picks up code/config changes)."""
        return self.load_tools(replace=True, reload=True)

    def register_tool(self, tool: BaseTool, *, replace: bool = False) -> BaseTool:
        """Register a hand-built tool instance (e.g. a ``RestEndpointTool``)."""
        return self.registry.register(tool, replace=replace)

    # -- introspection ------------------------------------------------------- #
    def list_tools(self, *, category: Optional[ToolCategory] = None, tag: Optional[str] = None) -> List[ToolSpec]:
        return self.registry.specs(category=category, tag=tag)

    def describe(self, name: str) -> Dict[str, Any]:
        """Human/JSON friendly description of one tool."""
        spec = self.registry.get(name).spec
        return {"name": spec.name, "description": spec.description, "category": spec.category.value,
                "version": spec.version, "tags": list(spec.tags),
                "required_permissions": sorted(spec.required_permissions),
                "parameters": spec.to_json_schema()}

    def stats(self) -> Dict[str, Any]:
        return {"tools": self.registry.names(), "calls": self.executor.stats()}

    # -- contexts ------------------------------------------------------------ #
    def new_context(self, **overrides: Any) -> ToolContext:
        """Create a context with the manager's default permissions and working dir."""
        ctx = ToolContext(permissions=set(self.default_permissions), working_dir=self.working_dir)
        return ctx.derive(**overrides) if overrides else ctx

    # -- execution ------------------------------------------------------------ #
    def call(self, name: str, arguments: Optional[Mapping[str, Any]] = None,
             context: Optional[ToolContext] = None, timeout: Optional[float] = None) -> ToolResult:
        """Execute a tool by name (never raises for tool failures)."""
        return self.executor.execute(name, arguments, context, timeout)

    def call_many(self, calls: Sequence[Tuple[str, Mapping[str, Any]]],
                  context: Optional[ToolContext] = None, timeout: Optional[float] = None) -> List[ToolResult]:
        return self.executor.execute_many(calls, context, timeout)

    async def acall(self, name: str, arguments: Optional[Mapping[str, Any]] = None,
                    context: Optional[ToolContext] = None, timeout: Optional[float] = None) -> ToolResult:
        return await self.executor.aexecute(name, arguments, context, timeout)

    # -- function calling ----------------------------------------------------- #
    def get_tool_definitions(self, provider: Union[str, Provider] = Provider.OPENAI, **filters: Any) -> List[Dict[str, Any]]:
        """Tool definitions in the provider's function-calling format."""
        return self.function_calling.get_tools(Provider(provider), **filters)

    def handle_model_response(self, response: Any, provider: Union[str, Provider] = Provider.OPENAI,
                              context: Optional[ToolContext] = None
                              ) -> Tuple[List[ToolCall], List[ToolResult], List[Dict[str, Any]]]:
        """Run any tool calls found in a model response; returns ``(calls, results, messages)``."""
        return self.function_calling.handle_response(response, Provider(provider), context)
