"""
feature_generator.py — Prompt-driven code generation, validated and staged.

Pipeline: prompt -> generated code -> syntax check -> security scan ->
quarantined staging for human review. There is deliberately no function
in this module that imports, exec()s, or hot-mounts generated code into
the running application.

Why that boundary exists:
    Auto-loading model-generated code into a live server collapses the
    gap between "text an LLM produced" and "code with full process
    privileges" to zero. That's risky even with a trusted model and a
    security scanner in front of it — scanners catch known-bad patterns,
    not novel logic bugs or subtly wrong behavior, and prompts feeding
    this pipeline may themselves be adversarial (e.g. if `prompt` ever
    originates from an end user rather than a trusted operator). The
    standard safe pattern — and the one this module implements — is:
    generate, validate, stage to a quarantined location for a human to
    review, and let integration happen through your normal code-review
    and deploy process. `FeatureStatus` stops at `PENDING_REVIEW`;
    nothing in this file can move a feature past that on its own.

Responsibilities:
    1. Turn a `FeatureRequest` (prompt + target module context) into a
       generated code string via a pluggable `CodeGenerationClient`.
    2. Validate syntax with `ast.parse`.
    3. Static-scan the AST for risky patterns (eval/exec, shell=True
       subprocess, unsafe deserialization, dynamic imports, broad
       excepts, obvious hardcoded secrets, outbound network calls) and
       produce a severity-scored report.
    4. Stage the code + its report to a pending-review directory with a
       unique id — never to an importable path — for a human to inspect
       and, separately, decide whether to merge.
"""

from __future__ import annotations

import ast
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

logger = logging.getLogger("app.feature_generator")


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------

class Severity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


_SEVERITY_ORDER: dict[Severity, int] = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}


class FeatureStatus(str, Enum):
    GENERATING = "generating"
    SYNTAX_INVALID = "syntax_invalid"
    PENDING_REVIEW = "pending_review"   # terminal state for this module
    REJECTED = "rejected"               # settable by a human reviewer, externally


# --------------------------------------------------------------------------
# Request / result models
# --------------------------------------------------------------------------

class FeatureRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    target_module: str = Field(
        description="Dotted path this code is intended for, e.g. 'services.notifications'. "
                     "Advisory only — used in the generation prompt, not a write target."
    )
    requested_by: str = Field(description="Identifier of the trusted operator making the request")
    context: dict[str, str] = Field(default_factory=dict)


class SecurityFinding(BaseModel):
    rule_id: str
    severity: Severity
    message: str
    line_number: int | None = None


class ValidationResult(BaseModel):
    syntax_valid: bool
    syntax_error: str | None = None
    findings: list[SecurityFinding] = Field(default_factory=list)
    highest_severity: Severity = Severity.INFO


class GeneratedFeature(BaseModel):
    id: str
    request: FeatureRequest
    code: str
    validation: ValidationResult
    status: FeatureStatus
    created_at: datetime
    staged_path: str | None = None


# --------------------------------------------------------------------------
# Pluggable code-generation backend
# --------------------------------------------------------------------------

class CodeGenerationClient(Protocol):
    """Implement against whatever model/API you use. Keeping this as a
    Protocol means feature_generator.py has no direct dependency on any
    specific LLM SDK or hardcoded credentials."""

    async def generate_code(self, prompt: str, context: dict[str, str]) -> str: ...


GENERATION_SYSTEM_PROMPT = """\
You write small, self-contained, asynchronous Python modules for a FastAPI \
backend. Rules:
- Return ONLY code — no prose, no markdown fences.
- Use async def for I/O-bound functions, full type hints, and Pydantic \
models for any structured data.
- Never use eval, exec, os.system, or subprocess with shell=True.
- Never hardcode secrets, API keys, or credentials — read them from \
function parameters or a passed-in config object.
- Prefer the standard library and already-common project dependencies \
(fastapi, pydantic, httpx) over introducing new dependencies.
"""


# --------------------------------------------------------------------------
# Security scanner (AST-based static analysis)
# --------------------------------------------------------------------------

class _SecurityVisitor(ast.NodeVisitor):
    """Walks the generated code's AST once, collecting findings. This is
    a heuristic pattern-matcher, not a full taint-tracking analyzer —
    treat CRITICAL/HIGH findings as near-certain blockers for auto-merge
    and everything else as "a human should look at this."""

    _DANGEROUS_CALLS: dict[str, Severity] = {
        "eval": Severity.CRITICAL,
        "exec": Severity.CRITICAL,
        "compile": Severity.HIGH,
        "__import__": Severity.HIGH,
    }

    _DANGEROUS_ATTR_CALLS: dict[tuple[str, str], Severity] = {
        ("os", "system"): Severity.CRITICAL,
        ("os", "popen"): Severity.CRITICAL,
        ("pickle", "load"): Severity.HIGH,
        ("pickle", "loads"): Severity.HIGH,
        ("yaml", "load"): Severity.MEDIUM,
        ("marshal", "loads"): Severity.HIGH,
    }

    _NETWORK_MODULES = {"socket", "requests", "httpx", "aiohttp", "urllib"}

    def __init__(self) -> None:
        self.findings: list[SecurityFinding] = []
        self._imported_modules: set[str] = set()

    def _add(self, rule_id: str, severity: Severity, message: str, node: ast.AST) -> None:
        self.findings.append(
            SecurityFinding(
                rule_id=rule_id,
                severity=severity,
                message=message,
                line_number=getattr(node, "lineno", None),
            )
        )

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._imported_modules.add(alias.name.split(".")[0])
            if alias.name.split(".")[0] in self._NETWORK_MODULES:
                self._add(
                    "network-import",
                    Severity.MEDIUM,
                    f"Imports networking module '{alias.name}' — confirm outbound calls are intended",
                    node,
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and node.module.split(".")[0] in self._NETWORK_MODULES:
            self._add(
                "network-import",
                Severity.MEDIUM,
                f"Imports from networking module '{node.module}' — confirm outbound calls are intended",
                node,
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name) and func.id in self._DANGEROUS_CALLS:
            self._add(
                f"dangerous-call-{func.id}",
                self._DANGEROUS_CALLS[func.id],
                f"Call to builtin '{func.id}' is disallowed in generated code",
                node,
            )

        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            key = (func.value.id, func.attr)
            if key in self._DANGEROUS_ATTR_CALLS:
                self._add(
                    f"dangerous-call-{key[0]}-{key[1]}",
                    self._DANGEROUS_ATTR_CALLS[key],
                    f"Call to '{key[0]}.{key[1]}' is disallowed or requires review",
                    node,
                )

            if func.attr == "run" and func.value.id == "subprocess":
                shell_true = any(
                    kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True
                    for kw in node.keywords
                )
                if shell_true:
                    self._add(
                        "subprocess-shell-true",
                        Severity.CRITICAL,
                        "subprocess.run with shell=True is disallowed",
                        node,
                    )
                else:
                    self._add(
                        "subprocess-usage",
                        Severity.MEDIUM,
                        "subprocess usage found — confirm this is intended and inputs are not user-controlled",
                        node,
                    )

        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is None:
            self._add(
                "bare-except",
                Severity.LOW,
                "Bare 'except:' can silently swallow errors — prefer catching specific exceptions",
                node,
            )
        self.generic_visit(node)


_SECRET_PATTERN = re.compile(
    r"""(?i)(api[_-]?key|secret|password|token)\s*=\s*['"][A-Za-z0-9_\-/+=]{8,}['"]"""
)


def _scan_for_hardcoded_secrets(code: str) -> list[SecurityFinding]:
    findings: list[SecurityFinding] = []
    for i, line in enumerate(code.splitlines(), start=1):
        if _SECRET_PATTERN.search(line):
            findings.append(
                SecurityFinding(
                    rule_id="hardcoded-secret",
                    severity=Severity.HIGH,
                    message="Line looks like a hardcoded credential — should come from config/env instead",
                    line_number=i,
                )
            )
    return findings


class SecurityScanner:
    def scan(self, code: str, tree: ast.AST) -> list[SecurityFinding]:
        visitor = _SecurityVisitor()
        visitor.visit(tree)
        return visitor.findings + _scan_for_hardcoded_secrets(code)


# --------------------------------------------------------------------------
# Staging store (quarantined — not an importable app path)
# --------------------------------------------------------------------------

class FeatureStagingStore:
    """Writes generated code + its validation report to a quarantine
    directory. This directory should NOT be on your app's import path or
    served by anything — it's a review inbox, not a deploy target."""

    def __init__(self, staging_dir: Path):
        self.staging_dir = staging_dir
        self.staging_dir.mkdir(parents=True, exist_ok=True)

    def save(self, feature: GeneratedFeature) -> Path:
        code_path = self.staging_dir / f"{feature.id}.py.txt"
        meta_path = self.staging_dir / f"{feature.id}.meta.json"

        code_path.write_text(feature.code, encoding="utf-8")
        meta_path.write_text(
            feature.model_dump_json(indent=2, exclude={"code"}),
            encoding="utf-8",
        )
        return code_path

    def load_meta(self, feature_id: str) -> dict:
        meta_path = self.staging_dir / f"{feature_id}.meta.json"
        return json.loads(meta_path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------

class FeatureGenerationError(Exception):
    """Raised when the generation backend itself fails (not a validation failure)."""


class FeatureGenerator:
    def __init__(
        self,
        client: CodeGenerationClient,
        staging_store: FeatureStagingStore,
        scanner: SecurityScanner | None = None,
    ):
        self._client = client
        self._staging_store = staging_store
        self._scanner = scanner or SecurityScanner()

    async def generate_feature(self, request: FeatureRequest) -> GeneratedFeature:
        feature_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)

        try:
            code = await self._client.generate_code(request.prompt, request.context)
        except Exception as exc:
            raise FeatureGenerationError("Code generation backend failed") from exc

        validation = self._validate(code)

        status = (
            FeatureStatus.SYNTAX_INVALID
            if not validation.syntax_valid
            else FeatureStatus.PENDING_REVIEW
        )

        feature = GeneratedFeature(
            id=feature_id,
            request=request,
            code=code,
            validation=validation,
            status=status,
            created_at=created_at,
        )

        if status is not FeatureStatus.SYNTAX_INVALID:
            path = self._staging_store.save(feature)
            feature.staged_path = str(path)
            logger.info(
                "feature_staged",
                extra={
                    "feature_id": feature_id,
                    "requested_by": request.requested_by,
                    "target_module": request.target_module,
                    "highest_severity": validation.highest_severity.value,
                    "finding_count": len(validation.findings),
                },
            )
        else:
            logger.warning(
                "feature_syntax_invalid",
                extra={"feature_id": feature_id, "requested_by": request.requested_by},
            )

        return feature

    def _validate(self, code: str) -> ValidationResult:
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return ValidationResult(
                syntax_valid=False,
                syntax_error=f"{exc.msg} (line {exc.lineno})",
            )

        findings = self._scanner.scan(code, tree)
        highest = max(
            (f.severity for f in findings), key=lambda s: _SEVERITY_ORDER[s], default=Severity.INFO
        )
        return ValidationResult(syntax_valid=True, findings=findings, highest_severity=highest)
