"""Pattern-based message router with middleware, timeouts and unroutable handling."""
from __future__ import annotations

import asyncio
import inspect
import itertools
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, FrozenSet, Iterable, List, Optional

from .communication_core import CommunicationError, CommunicationTimeoutError, RoutingError
from .messaging.event_bus import topic_matches, validate_topic
from .protocol_handler import Envelope, MessageType

logger = logging.getLogger(__name__)


@dataclass
class RouteContext:
    """Per-dispatch context handed to handlers and middleware."""

    connection: Any = None
    principal: Any = None
    attributes: Dict[str, Any] = field(default_factory=dict)


Handler = Callable[[Envelope, RouteContext], Any]  # sync or async
Next = Callable[[], Awaitable[Any]]
Middleware = Callable[[Envelope, RouteContext, Next], Awaitable[Any]]
UnroutableHook = Callable[[Envelope], Any]


@dataclass(frozen=True, eq=False)
class Route:
    pattern: str
    handler: Handler
    types: FrozenSet[MessageType]
    priority: int
    timeout_s: Optional[float]
    order: int

    @property
    def sort_key(self) -> tuple:
        wildcards = sum(2 if s == "#" else 1 for s in self.pattern.split(".") if s in ("*", "#"))
        return (-self.priority, wildcards, self.order)


class MessageRouter:
    """Routes an :class:`Envelope` to the best-matching handler.

    Precedence: higher ``priority`` first, then fewer wildcards, then registration
    order. Only the first matching route runs (request semantics); use the event
    manager for fan-out.
    """

    def __init__(self, *, default_timeout_s: Optional[float] = 30.0) -> None:
        self._routes: List[Route] = []
        self._middleware: List[Middleware] = []
        self._default: Optional[Handler] = None
        self._on_unroutable: Optional[UnroutableHook] = None
        self._default_timeout = default_timeout_s
        self._counter = itertools.count()
        self._stats: Dict[str, int] = {"routed": 0, "unroutable": 0, "errors": 0, "timeouts": 0}
        self._per_route: Dict[str, int] = {}

    # -- configuration ------------------------------------------------------ #
    def add_route(self, pattern: str, handler: Handler, *,
                  types: Iterable[MessageType] = (MessageType.REQUEST,), priority: int = 0,
                  timeout_s: Optional[float] = None) -> Route:
        validate_topic(pattern, allow_wildcards=True)
        route = Route(pattern, handler, frozenset(types), priority, timeout_s, next(self._counter))
        self._routes.append(route)
        self._routes.sort(key=lambda r: r.sort_key)
        return route

    def on(self, pattern: str, **kwargs: Any) -> Callable[[Handler], Handler]:
        """Decorator form of :meth:`add_route`."""
        def deco(fn: Handler) -> Handler:
            self.add_route(pattern, fn, **kwargs)
            return fn
        return deco

    def remove_route(self, route: Route) -> None:
        self._routes = [r for r in self._routes if r is not route]

    def use(self, middleware: Middleware) -> None:
        self._middleware.append(middleware)

    def set_default(self, handler: Optional[Handler]) -> None:
        self._default = handler

    def on_unroutable(self, hook: Optional[UnroutableHook]) -> None:
        """Dead-letter hook invoked for envelopes with no matching route."""
        self._on_unroutable = hook

    # -- dispatch ----------------------------------------------------------- #
    async def route(self, env: Envelope, context: Optional[RouteContext] = None) -> Any:
        ctx = context if isinstance(context, RouteContext) else RouteContext()
        matched = next((r for r in self._routes if env.type in r.types and topic_matches(r.pattern, env.topic)), None)
        handler: Optional[Handler] = matched.handler if matched else self._default
        if handler is None:
            self._stats["unroutable"] += 1
            await self._notify_unroutable(env)
            raise RoutingError(f"no route for {env.type.value} '{env.topic}'")
        timeout = matched.timeout_s if matched and matched.timeout_s is not None else self._default_timeout

        async def terminal() -> Any:
            res = handler(env, ctx)
            return await res if inspect.isawaitable(res) else res

        call: Next = terminal
        for mw in reversed(self._middleware):
            call = self._link(mw, env, ctx, call)
        try:
            result = await (asyncio.wait_for(call(), timeout) if timeout else call())
        except asyncio.TimeoutError:
            self._stats["timeouts"] += 1
            raise CommunicationTimeoutError(f"handler for '{env.topic}' timed out") from None
        except CommunicationError:
            self._stats["errors"] += 1
            raise
        except Exception as exc:
            self._stats["errors"] += 1
            logger.exception("Handler for %s failed", env.topic)
            raise CommunicationError("handler failed", code="handler_error") from exc
        self._stats["routed"] += 1
        key = matched.pattern if matched else "<default>"
        self._per_route[key] = self._per_route.get(key, 0) + 1
        return result

    def stats(self) -> Dict[str, Any]:
        return {**self._stats, "routes": len(self._routes), "per_route": dict(self._per_route)}

    # -- internals ---------------------------------------------------------- #
    @staticmethod
    def _link(mw: Middleware, env: Envelope, ctx: RouteContext, nxt: Next) -> Next:
        async def run() -> Any:
            return await mw(env, ctx, nxt)
        return run

    async def _notify_unroutable(self, env: Envelope) -> None:
        if self._on_unroutable is None:
            return
        try:
            res = self._on_unroutable(env)
            if inspect.isawaitable(res):
                await res
        except Exception:
            logger.exception("unroutable hook failed")


async def logging_middleware(env: Envelope, ctx: RouteContext, call_next: Next) -> Any:
    """Example middleware: logs topic, outcome and duration."""
    start = time.perf_counter()
    try:
        result = await call_next()
        logger.info("route %s ok in %.1fms", env.topic, (time.perf_counter() - start) * 1000)
        return result
    except Exception as exc:
        logger.warning("route %s failed after %.1fms: %r", env.topic, (time.perf_counter() - start) * 1000, exc)
        raise
