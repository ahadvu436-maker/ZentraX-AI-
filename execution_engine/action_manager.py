"""Registry of named, reusable actions."""
from __future__ import annotations

import inspect
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .execution_core import ActionArgumentError, ActionError, ActionNotFoundError, TaskSpec
from .retry_manager import RetryPolicy
from .sandbox.resource_limits import ResourceLimits

__all__ = ["Action", "ActionManager"]

_NAME_RE = re.compile(r"^[A-Za-z_][\w.\-:]{0,127}$")


@dataclass(frozen=True)
class Action:
    """A registered callable plus its default execution options."""

    name: str
    func: Callable[..., Any]
    description: str = ""
    timeout: Optional[float] = None
    retry_policy: Optional[RetryPolicy] = None
    sandboxed: Optional[bool] = None
    limits: Optional[ResourceLimits] = None
    tags: Tuple[str, ...] = ()
    is_async: bool = field(default=False)


class ActionManager:
    """Thread-safe registry mapping action names to callables.

    Actions run through :class:`ExecutionCore` (``core.run_action``), which gives
    them retries, monitoring and optional sandboxing for free.
    """

    def __init__(self) -> None:
        self._actions: Dict[str, Action] = {}
        self._lock = threading.RLock()

    def register(
        self, name: str, func: Callable[..., Any], *, description: str = "",
        timeout: Optional[float] = None, retry_policy: Optional[RetryPolicy] = None,
        sandboxed: Optional[bool] = None, limits: Optional[ResourceLimits] = None,
        tags: Sequence[str] = (), replace: bool = False,
    ) -> Action:
        if not isinstance(name, str) or not _NAME_RE.match(name):
            raise ActionError(f"invalid action name: {name!r}")
        if not callable(func):
            raise ActionError(f"action {name!r}: func must be callable")
        action = Action(
            name=name, func=func, description=description or (inspect.getdoc(func) or "").split("\n")[0],
            timeout=timeout, retry_policy=retry_policy, sandboxed=sandboxed, limits=limits,
            tags=tuple(tags), is_async=inspect.iscoroutinefunction(func))
        with self._lock:
            if name in self._actions and not replace:
                raise ActionError(f"action {name!r} already registered (use replace=True)")
            self._actions[name] = action
        return action

    def action(self, name: Optional[str] = None, **options: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator form of :meth:`register`."""
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.register(name or func.__name__, func, **options)
            return func
        return decorator

    def unregister(self, name: str) -> None:
        with self._lock:
            if self._actions.pop(name, None) is None:
                raise ActionNotFoundError(f"unknown action {name!r}")

    def has(self, name: str) -> bool:
        with self._lock:
            return name in self._actions

    def get(self, name: str) -> Action:
        with self._lock:
            try:
                return self._actions[name]
            except KeyError:
                raise ActionNotFoundError(f"unknown action {name!r}") from None

    def list(self, tag: Optional[str] = None) -> List[Action]:
        with self._lock:
            actions = sorted(self._actions.values(), key=lambda a: a.name)
        return [a for a in actions if tag is None or tag in a.tags]

    def validate_arguments(self, name: str, args: Sequence[Any], kwargs: Mapping[str, Any]) -> None:
        """Raise :class:`ActionArgumentError` if the arguments don't fit the signature."""
        action = self.get(name)
        try:
            inspect.signature(action.func).bind(*args, **kwargs)
        except TypeError as exc:
            raise ActionArgumentError(f"action {name!r}: {exc}") from exc
        except ValueError:  # signature unavailable (some builtins)
            pass

    def build_spec(self, name: str, args: Sequence[Any] = (),
                   kwargs: Optional[Mapping[str, Any]] = None, **overrides: Any) -> TaskSpec:
        """Create a validated :class:`TaskSpec`; ``overrides`` replace the action's defaults."""
        action = self.get(name)
        kwargs = dict(kwargs or {})
        self.validate_arguments(name, args, kwargs)
        options: Dict[str, Any] = dict(
            timeout=action.timeout, retry_policy=action.retry_policy,
            sandboxed=action.sandboxed, limits=action.limits,
            metadata={"action": action.name})
        options.update(overrides)
        return TaskSpec(func=action.func, args=tuple(args), kwargs=kwargs, name=action.name, **options)
