"""Function-calling bridge between LLM providers and the tool system.

* Render tool specs in provider formats (OpenAI, Anthropic, Gemini, generic JSON Schema).
* Parse model responses into provider-neutral :class:`ToolCall` objects.
* Execute calls through the :class:`ToolExecutor` (in parallel) and format the results as the
  messages each provider expects for the next turn.

Responses may be plain dicts or SDK objects (anything with ``model_dump()``/``to_dict()``).
"""
from __future__ import annotations

import enum
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .tool_core import ToolContext, ToolResult, ToolValidationError
from .tool_executor import ToolExecutor
from .tool_registry import ToolRegistry

logger = logging.getLogger(__name__)

__all__ = ["Provider", "ToolCall", "FunctionCallingAdapter"]


class Provider(str, enum.Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    GENERIC = "generic"


@dataclass
class ToolCall:
    """A tool invocation requested by a model."""

    id: str
    name: str
    arguments: Dict[str, Any]
    parse_error: Optional[str] = None  # set when the model's arguments were not valid JSON


def _as_dict(obj: Any) -> Any:
    """Normalise SDK objects to plain dicts/lists."""
    if isinstance(obj, (dict, list, str, int, float, bool)) or obj is None:
        return obj
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:  # noqa: BLE001
                continue
    return getattr(obj, "__dict__", obj)


def _parse_args(raw: Any) -> Tuple[Dict[str, Any], Optional[str]]:
    if raw is None or raw == "":
        return {}, None
    if isinstance(raw, Mapping):
        return dict(raw), None
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except ValueError as exc:
            return {}, f"arguments are not valid JSON: {exc}"
        if isinstance(value, dict):
            return value, None
        return {}, "arguments must be a JSON object"
    return {}, f"unsupported arguments type {type(raw).__name__}"


class FunctionCallingAdapter:
    """Glue between a registry/executor pair and LLM function-calling protocols."""

    def __init__(self, registry: ToolRegistry, executor: ToolExecutor,
                 default_provider: Provider = Provider.OPENAI, max_result_chars: int = 20_000) -> None:
        self.registry = registry
        self.executor = executor
        self.default_provider = Provider(default_provider)
        self.max_result_chars = max_result_chars

    # -- schemas ------------------------------------------------------------- #
    def get_tools(self, provider: Optional[Provider] = None, *, names: Optional[Sequence[str]] = None,
                  **filters: Any) -> List[Dict[str, Any]]:
        """Tool definitions for ``provider`` (optionally restricted to ``names``)."""
        prov = Provider(provider or self.default_provider)
        specs = [t.spec for t in self.registry.list(**filters) if names is None or t.spec.name in names]
        if prov is Provider.OPENAI:
            return [{"type": "function", "function": {"name": s.name, "description": s.description,
                                                       "parameters": s.to_json_schema()}} for s in specs]
        if prov is Provider.ANTHROPIC:
            return [{"name": s.name, "description": s.description,
                     "input_schema": s.to_json_schema()} for s in specs]
        if prov is Provider.GEMINI:
            return [{"functionDeclarations": [
                {"name": s.name, "description": s.description,
                 "parameters": {k: v for k, v in s.to_json_schema().items() if k != "additionalProperties"}}
                for s in specs]}] if specs else []
        return [{"name": s.name, "description": s.description, "parameters": s.to_json_schema()}
                for s in specs]

    # -- parsing ------------------------------------------------------------- #
    def parse_tool_calls(self, response: Any, provider: Optional[Provider] = None) -> List[ToolCall]:
        """Extract tool calls from a provider response (empty list if the model made none)."""
        prov = Provider(provider or self.default_provider)
        data = _as_dict(response)
        calls: List[ToolCall] = []
        if prov is Provider.OPENAI:
            msg = data
            if isinstance(data, dict) and data.get("choices"):
                msg = _as_dict(data["choices"][0]).get("message", {})
            msg = _as_dict(msg) or {}
            for tc in msg.get("tool_calls") or []:
                tc = _as_dict(tc)
                fn = _as_dict(tc.get("function") or {})
                args, err = _parse_args(fn.get("arguments"))
                calls.append(ToolCall(tc.get("id") or self._new_id(), fn.get("name", ""), args, err))
            legacy = msg.get("function_call")
            if legacy and not calls:
                legacy = _as_dict(legacy)
                args, err = _parse_args(legacy.get("arguments"))
                calls.append(ToolCall(self._new_id(), legacy.get("name", ""), args, err))
        elif prov is Provider.ANTHROPIC:
            for block in (data or {}).get("content") or []:
                block = _as_dict(block)
                if block.get("type") == "tool_use":
                    args, err = _parse_args(block.get("input"))
                    calls.append(ToolCall(block.get("id") or self._new_id(), block.get("name", ""), args, err))
        elif prov is Provider.GEMINI:
            for cand in (data or {}).get("candidates") or []:
                for part in (_as_dict(_as_dict(cand).get("content") or {}).get("parts") or []):
                    fc = _as_dict(part).get("functionCall") or _as_dict(part).get("function_call")
                    if fc:
                        fc = _as_dict(fc)
                        args, err = _parse_args(fc.get("args"))
                        calls.append(ToolCall(self._new_id(), fc.get("name", ""), args, err))
        else:
            items = data if isinstance(data, list) else [data]
            for item in items:
                item = _as_dict(item)
                if isinstance(item, dict) and item.get("name"):
                    args, err = _parse_args(item.get("arguments", item.get("args", item.get("input"))))
                    calls.append(ToolCall(item.get("id") or self._new_id(), item["name"], args, err))
        return calls

    # -- execution ------------------------------------------------------------ #
    def execute_calls(self, calls: Sequence[ToolCall], context: Optional[ToolContext] = None,
                      timeout: Optional[float] = None) -> List[ToolResult]:
        """Run calls in parallel through the executor; one result per call, in order."""
        results: List[Optional[ToolResult]] = [None] * len(calls)
        runnable: List[Tuple[int, ToolCall]] = []
        for i, call in enumerate(calls):
            if call.parse_error:
                results[i] = ToolResult.fail(call.name or "unknown", ToolValidationError(call.parse_error))
            elif not call.name:
                results[i] = ToolResult.fail("unknown", ToolValidationError("tool call has no name"))
            else:
                runnable.append((i, call))
        executed = self.executor.execute_many([(c.name, c.arguments) for _, c in runnable],
                                              context=context, timeout=timeout)
        for (i, _), res in zip(runnable, executed):
            results[i] = res
        return [r for r in results if r is not None]

    # -- formatting ----------------------------------------------------------- #
    def format_results(self, calls: Sequence[ToolCall], results: Sequence[ToolResult],
                       provider: Optional[Provider] = None) -> List[Dict[str, Any]]:
        """Messages to append to the conversation so the model can see the tool outputs."""
        prov = Provider(provider or self.default_provider)
        pairs = list(zip(calls, results))
        text = lambda r: r.to_message(self.max_result_chars)  # noqa: E731
        if prov is Provider.OPENAI:
            return [{"role": "tool", "tool_call_id": c.id, "content": text(r)} for c, r in pairs]
        if prov is Provider.ANTHROPIC:
            return [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": c.id, "content": text(r), "is_error": not r.success}
                for c, r in pairs]}] if pairs else []
        if prov is Provider.GEMINI:
            return [{"role": "user", "parts": [
                {"functionResponse": {"name": c.name, "response": {
                    "result" if r.success else "error": r.output if r.success else r.error}}}
                for c, r in pairs]}] if pairs else []
        return [{"id": c.id, "name": c.name, **r.to_dict()} for c, r in pairs]

    def handle_response(self, response: Any, provider: Optional[Provider] = None,
                        context: Optional[ToolContext] = None
                        ) -> Tuple[List[ToolCall], List[ToolResult], List[Dict[str, Any]]]:
        """Parse -> execute -> format in one call.

        Returns ``(calls, results, messages)``; append ``messages`` to the conversation and call
        the model again.  If ``calls`` is empty the model produced a final answer.
        """
        calls = self.parse_tool_calls(response, provider)
        if not calls:
            return [], [], []
        results = self.execute_calls(calls, context)
        return calls, results, self.format_results(calls, results, provider)

    @staticmethod
    def _new_id() -> str:
        return "call_" + uuid.uuid4().hex[:16]
