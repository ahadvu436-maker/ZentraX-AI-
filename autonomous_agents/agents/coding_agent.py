"""CodingAgent: code generation, static review, validation and (opt-in) execution.

Payload actions
---------------
``validate``  syntax-check Python                 keys: ``code``
``review``    AST-based static analysis           keys: ``code`` (or upstream output / ``context["artifact"]``)
``generate``  LLM code generation (needs ``llm``) keys: ``spec``, ``language``, ``existing_code``
``execute``   run Python in a subprocess          keys: ``code``, ``timeout``, ``stdin``
              (only when ``allow_execution=True``; NOT a security sandbox)
"""

from __future__ import annotations

import ast
import asyncio
import os
import re
import sys
import tempfile
import time
from typing import Any, Dict, List, Mapping, Optional

from ..agent_core import AgentConfig, BaseAgent, Reasoner, Task, call_reasoner

__all__ = ["CodingAgent", "analyze_python"]

_FENCE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)```", re.DOTALL)
_BRANCHES = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler, ast.BoolOp, ast.IfExp, ast.comprehension)
_WEIGHTS = {"error": 15, "warning": 5, "info": 1}


def _extract_code(text: str) -> str:
    blocks = _FENCE.findall(text)
    return (max(blocks, key=len) if blocks else text).strip("\n")


def _syntax_problem(code: str) -> Optional[str]:
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return f"line {exc.lineno}: {exc.msg}"
    return None


def analyze_python(code: str) -> Dict[str, Any]:
    """Static analysis of Python source. Raises ``SyntaxError`` on invalid code."""
    tree = ast.parse(code)
    issues: List[Dict[str, Any]] = []

    def add(node: ast.AST, severity: str, rule: str, message: str) -> None:
        issues.append({"line": getattr(node, "lineno", 0), "severity": severity, "rule": rule, "message": message})

    functions = classes = 0
    complexities: List[int] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions += 1
            length = (node.end_lineno or node.lineno) - node.lineno + 1
            complexity = 1 + sum(isinstance(n, _BRANCHES) for n in ast.walk(node))
            complexities.append(complexity)
            a = node.args
            params = len(a.posonlyargs) + len(a.args) + len(a.kwonlyargs) + bool(a.vararg) + bool(a.kwarg)
            if length > 50:
                add(node, "warning", "long-function", f"'{node.name}' is {length} lines long")
            if complexity > 10:
                add(node, "warning", "complexity", f"'{node.name}' has cyclomatic complexity {complexity}")
            if params > 7:
                add(node, "warning", "many-params", f"'{node.name}' takes {params} parameters")
            for default in list(a.defaults) + [d for d in a.kw_defaults if d is not None]:
                if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                    add(default, "error", "mutable-default", f"'{node.name}' uses a mutable default argument")
            if not node.name.startswith("_") and not ast.get_docstring(node):
                add(node, "info", "missing-docstring", f"public function '{node.name}' has no docstring")
        elif isinstance(node, ast.ClassDef):
            classes += 1
        elif isinstance(node, ast.ExceptHandler) and node.type is None:
            add(node, "warning", "bare-except", "bare 'except:' swallows all exceptions")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("eval", "exec"):
            add(node, "error", "dangerous-call", f"use of {node.func.id}() is a security risk")
        elif isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names):
            add(node, "warning", "wildcard-import", f"wildcard import from '{node.module}'")

    issues.sort(key=lambda i: (i["line"], i["rule"]))
    penalty = sum(_WEIGHTS[i["severity"]] for i in issues)
    return {
        "valid": True,
        "metrics": {
            "lines": len(code.splitlines()),
            "functions": functions,
            "classes": classes,
            "avg_complexity": round(sum(complexities) / len(complexities), 2) if complexities else 0.0,
            "max_complexity": max(complexities, default=0),
        },
        "issues": issues,
        "score": max(0, 100 - penalty),
    }


class CodingAgent(BaseAgent):
    agent_type = "coding"
    DEFAULT_CAPABILITIES = frozenset({"coding", "code_generation", "code_review", "static_analysis"})

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        *,
        agent_id: Optional[str] = None,
        llm: Optional[Reasoner] = None,
        allow_execution: bool = False,
        max_generation_attempts: int = 3,
        execution_timeout: float = 10.0,
        max_output_chars: int = 20_000,
    ) -> None:
        caps = set(self.DEFAULT_CAPABILITIES)
        if allow_execution:
            caps.add("code_execution")
        config = config or AgentConfig(name="coding-agent", capabilities=caps, max_concurrent_tasks=2)
        super().__init__(config, agent_id)
        self._llm = llm
        self._allow_execution = allow_execution
        self._max_attempts = max(1, max_generation_attempts)
        self._exec_timeout = execution_timeout
        self._max_output = max_output_chars

    async def handle_task(self, task: Task) -> Dict[str, Any]:
        action = str(task.payload.get("action", "review")).lower()
        if action == "validate":
            code = self._resolve_code(task)
            problem = await asyncio.to_thread(_syntax_problem, code)
            return {"valid": problem is None, "error": problem}
        if action == "review":
            code = self._resolve_code(task)
            try:
                return await asyncio.to_thread(analyze_python, code)
            except SyntaxError as exc:
                return {"valid": False, "error": f"line {exc.lineno}: {exc.msg}", "issues": [], "score": 0}
        if action in ("generate", "refactor"):
            return await self._generate(task)
        if action == "execute":
            return await self._execute(task)
        raise ValueError(f"unknown coding action: {action!r}")

    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_code(task: Task) -> str:
        """Find code in payload, ``context['artifact']`` or upstream dependency outputs."""

        def pick(value: Any) -> Optional[str]:
            if isinstance(value, str) and value.strip():
                return value
            if isinstance(value, Mapping) and isinstance(value.get("code"), str) and value["code"].strip():
                return value["code"]
            return None

        code = pick(task.payload.get("code")) or pick(task.context.get("artifact"))
        if code is None:
            for upstream in reversed(list((task.context.get("dependency_results") or {}).values())):
                code = pick(upstream)
                if code:
                    break
        if code is None:
            code = pick(task.context.get("previous_output"))
        if code is None:
            raise ValueError("no code found in payload, context artifact or upstream results")
        return code

    async def _generate(self, task: Task) -> Dict[str, Any]:
        if self._llm is None:
            raise RuntimeError("code generation requires an 'llm' to be configured on CodingAgent")
        p = task.payload
        spec = p.get("spec") or p.get("instructions") or p.get("goal") or task.description
        if not spec:
            raise ValueError("generate requires a 'spec'")
        language = str(p.get("language", "python")).lower()
        existing = p.get("existing_code")
        feedback = task.context.get("feedback")
        prompt = f"Write {language} code for the following requirement. Return only one fenced code block.\n\n{spec}\n"
        if existing:
            prompt += f"\nModify this existing code:\n```{language}\n{existing}\n```\n"
        if feedback:
            prompt += f"\nAddress this reviewer feedback:\n{feedback}\n"

        base_prompt, code, problem = prompt, "", None
        for attempt in range(1, self._max_attempts + 1):
            code = _extract_code(await call_reasoner(self._llm, prompt))
            if language != "python":
                return {"language": language, "code": code, "attempts": attempt, "validated": False}
            problem = _syntax_problem(code)
            if problem is None:
                result: Dict[str, Any] = {"language": language, "code": code, "attempts": attempt, "validated": True}
                if p.get("review", True):
                    result["review"] = await asyncio.to_thread(analyze_python, code)
                return result
            prompt = (
                f"{base_prompt}\nYour previous attempt had a syntax error ({problem}). "
                f"Return the corrected complete code.\n```python\n{code}\n```"
            )
        raise RuntimeError(f"no syntactically valid code after {self._max_attempts} attempts: {problem}")

    async def _execute(self, task: Task) -> Dict[str, Any]:
        if not self._allow_execution:
            raise PermissionError("code execution is disabled (construct CodingAgent with allow_execution=True)")
        code = self._resolve_code(task)
        problem = _syntax_problem(code)
        if problem:
            raise ValueError(f"cannot execute invalid Python: {problem}")
        timeout = float(task.payload.get("timeout", self._exec_timeout))
        stdin = str(task.payload.get("stdin", "")).encode()
        env = {k: os.environ[k] for k in ("PATH", "SYSTEMROOT") if k in os.environ}
        env["PYTHONIOENCODING"] = "utf-8"
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="zentrax-exec-") as cwd:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-I",
                "-c",
                code,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )
            timed_out = False
            try:
                out, err = await asyncio.wait_for(proc.communicate(stdin), timeout=timeout)
            except asyncio.TimeoutError:
                timed_out = True
                proc.kill()
                out, err = await proc.communicate()
        return {
            "ok": proc.returncode == 0 and not timed_out,
            "returncode": proc.returncode,
            "timed_out": timed_out,
            "stdout": out.decode(errors="replace")[: self._max_output],
            "stderr": err.decode(errors="replace")[: self._max_output],
            "duration": round(time.monotonic() - started, 4),
        }
