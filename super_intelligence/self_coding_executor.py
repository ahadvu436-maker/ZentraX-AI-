#!/usr/bin/env python3
"""
ZentraX-AI  |  super_intelligence/self_coding_executor.py

A local-only, human-gated pipeline for analyzing a codebase, validating
machine-written patches, and applying them safely.

Pipeline
    1. ANALYZE   AST scan of the project (structure, complexity, risk findings)
    2. PROPOSE   a PatchProvider (JSON file or a *local* model command) yields a
                 PatchProposal: search/replace edits and/or new files
    3. VALIDATE  path confinement, protected files, size limits, AST safety scan
                 (only *newly introduced* risky constructs are rejected)
    4. SANDBOX   proposal is applied to a throw-away COPY of the project, then
                 py_compile + the test suite run there (no network namespace when
                 the OS supports it, CPU/memory/file-size limits, scrubbed env)
    5. APPROVE   unified diff shown; a human (or an explicit --yes) must confirm
    6. COMMIT    originals backed up, atomic writes, hash verification,
                 automatic rollback on any failure
    7. AUDIT     hash-chained JSONL log; every run is replayable via `rollback`

Security model (defense in depth, NOT a perfect sandbox)
    * Never executes proposed code in the real tree; only in a temp copy.
    * No network access in the executor itself; providers must be local.
    * Never deletes project files; protected paths (this file, secrets, .git,
      security/config modules) can never be modified.
    * For strong isolation of the test run, execute this tool inside a container
      or VM, and use --require-isolation to fail closed when the OS cannot
      provide a network namespace.

Patch file format (JSON)
    {
      "title": "Add retry helper",
      "rationale": "Why this change is needed",
      "changes": [
        {"path": "backend/app/utils.py", "action": "modify",
         "edits": [{"search": "old text (must match exactly once)",
                    "replace": "new text"}]},
        {"path": "backend/app/retry.py", "action": "create",
         "content": "def retry(...): ..."}
      ]
    }

CLI
    python self_coding_executor.py analyze [--json]
    python self_coding_executor.py check    patch.json
    python self_coding_executor.py apply    patch.json [--yes]
    python self_coding_executor.py generate --goal "..." --model-cmd "ollama run codellama" --out patch.json
    python self_coding_executor.py rollback <patch-id> [--force]
    python self_coding_executor.py history
    python self_coding_executor.py verify-log
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import difflib
import functools
import hashlib
import json
import os
import py_compile
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable, Optional, Protocol

# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class ExecutorError(Exception):
    """Base class for all executor failures."""


class SecurityViolation(ExecutorError):
    """A proposal tried to leave its allowed boundaries."""


class ValidationError(ExecutorError):
    """A proposal is malformed or fails validation."""


class SandboxFailure(ExecutorError):
    """Sandboxed checks (compile/tests) failed."""


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ExecutorConfig:
    root: Path
    state_dirname: str = ".zentrax"
    editable_suffixes: frozenset = frozenset(
        {".py", ".md", ".json", ".toml", ".yaml", ".yml", ".txt", ".cfg"}
    )
    # Files that can never be modified by generated patches.
    protected_paths: frozenset = frozenset(
        {
            "super_intelligence/self_coding_executor.py",
            "backend/app/core/security.py",
            "backend/app/core/config.py",
        }
    )
    protected_names: frozenset = frozenset({"id_rsa", "id_ed25519", "credentials.json"})
    protected_suffixes: frozenset = frozenset({".pem", ".key", ".crt", ".p12", ".pfx"})
    protected_parts: frozenset = frozenset({".git", ".zentrax", ".venv", "venv", "node_modules"})
    ignore_dirs: frozenset = frozenset(
        {".git", ".zentrax", ".venv", "venv", "node_modules", "__pycache__",
         ".mypy_cache", ".pytest_cache", "dist", "build"}
    )
    max_files_per_patch: int = 10
    max_file_bytes: int = 200_000
    forbidden_modules: frozenset = frozenset(
        {"socket", "ssl", "requests", "urllib", "urllib3", "http", "httpx", "aiohttp",
         "ftplib", "smtplib", "telnetlib", "xmlrpc", "subprocess", "ctypes", "pickle", "marshal"}
    )
    forbidden_calls: frozenset = frozenset({"eval", "exec", "compile", "__import__", "breakpoint"})
    forbidden_dotted: frozenset = frozenset(
        {"os.system", "os.popen", "os.remove", "os.unlink", "os.rmdir", "os.removedirs",
         "shutil.rmtree", "importlib.import_module"}
    )
    forbidden_attrs: frozenset = frozenset({"__subclasses__", "__globals__", "__builtins__"})
    test_command: Optional[tuple] = None  # auto-detected when None
    test_timeout: int = 300
    memory_limit_mb: int = 2048
    require_isolation: bool = False  # fail closed if no network namespace

    @property
    def state_dir(self) -> Path:
        return self.root / self.state_dirname


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _dotted(node: ast.AST) -> Optional[str]:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _atomic_write(target: Path, text: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8", newline="")
        if target.exists():
            shutil.copymode(target, tmp)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Path confinement
# --------------------------------------------------------------------------- #


class PathGuard:
    def __init__(self, cfg: ExecutorConfig):
        self.cfg = cfg
        self.root = cfg.root.resolve()

    def resolve(self, rel: str) -> Path:
        """Resolve `rel` strictly inside the project root (no symlinks, no '..')."""
        if not isinstance(rel, str) or not rel or "\x00" in rel:
            raise SecurityViolation(f"invalid path: {rel!r}")
        p = PurePosixPath(rel.replace("\\", "/"))
        if p.is_absolute() or ".." in p.parts:
            raise SecurityViolation(f"path escapes project root: {rel}")
        target = (self.root / Path(*p.parts)).resolve()
        if not target.is_relative_to(self.root):
            raise SecurityViolation(f"path escapes project root: {rel}")
        cur = self.root
        for part in p.parts:
            cur = cur / part
            if cur.is_symlink():
                raise SecurityViolation(f"symlinks are not allowed: {rel}")
        return target

    def check_editable(self, rel: str) -> tuple:
        target = self.resolve(rel)
        relp = target.relative_to(self.root).as_posix()
        c = self.cfg
        if (set(relp.split("/")) & c.protected_parts
                or relp in c.protected_paths
                or target.name.startswith(".env")
                or target.name in c.protected_names
                or target.suffix in c.protected_suffixes):
            raise SecurityViolation(f"protected path: {relp}")
        if target.suffix not in c.editable_suffixes:
            raise SecurityViolation(f"file type not editable: {relp}")
        return target, relp


# --------------------------------------------------------------------------- #
# Static safety scanner
# --------------------------------------------------------------------------- #


class SafetyScanner:
    """AST-based scanner. Defense in depth only; it is not a sandbox."""

    def __init__(self, cfg: ExecutorConfig):
        self.cfg = cfg

    def scan(self, source: str, filename: str = "<patch>") -> list:
        """Return a list of (key, message). `key` is line-independent."""
        try:
            tree = ast.parse(source, filename)
        except SyntaxError as e:
            return [("syntax", f"{filename}:{e.lineno}: syntax error: {e.msg}")]
        c, out = self.cfg, []
        for n in ast.walk(tree):
            line = getattr(n, "lineno", 0)
            if isinstance(n, ast.Import):
                for a in n.names:
                    top = a.name.split(".")[0]
                    if top in c.forbidden_modules:
                        out.append((f"import:{top}", f"{filename}:{line}: import of '{top}'"))
            elif isinstance(n, ast.ImportFrom) and n.module:
                top = n.module.split(".")[0]
                if top in c.forbidden_modules:
                    out.append((f"import:{top}", f"{filename}:{line}: import from '{top}'"))
            elif isinstance(n, ast.Call):
                name = _dotted(n.func)
                if name and (name in c.forbidden_calls or name in c.forbidden_dotted
                             or name.startswith(("os.exec", "os.spawn"))):
                    out.append((f"call:{name}", f"{filename}:{line}: call to '{name}'"))
            elif isinstance(n, ast.Attribute) and n.attr in c.forbidden_attrs:
                out.append((f"attr:{n.attr}", f"{filename}:{line}: access to '{n.attr}'"))
        return out


# --------------------------------------------------------------------------- #
# Codebase analysis
# --------------------------------------------------------------------------- #

_BRANCH = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.ExceptHandler,
           ast.BoolOp, ast.IfExp, ast.comprehension)


class CodebaseAnalyzer:
    def __init__(self, cfg: ExecutorConfig, scanner: SafetyScanner):
        self.cfg, self.scanner = cfg, scanner
        self.root = cfg.root.resolve()

    def _iter_py(self):
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames
                           if d not in self.cfg.ignore_dirs and not (Path(dirpath) / d).is_symlink()]
            for f in filenames:
                p = Path(dirpath) / f
                if f.endswith(".py") and not p.is_symlink():
                    yield p

    def analyze(self) -> dict:
        files, test_stems = [], set()
        for p in self._iter_py():
            rel = p.relative_to(self.root).as_posix()
            if p.name.startswith("test_"):
                test_stems.add(p.stem[5:])
            try:
                raw = p.read_bytes()
                if len(raw) > 1_000_000:
                    continue
                src = raw.decode("utf-8")
            except (OSError, UnicodeDecodeError):
                files.append({"path": rel, "error": "unreadable"})
                continue
            entry = {"path": rel, "sha256": _sha(raw), "loc": src.count("\n") + 1,
                     "functions": [], "classes": [], "risk": []}
            try:
                tree = ast.parse(src, rel)
            except SyntaxError as e:
                entry["error"] = f"syntax error line {e.lineno}"
                files.append(entry)
                continue
            for n in ast.walk(tree):
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    entry["functions"].append({
                        "name": n.name, "line": n.lineno,
                        "complexity": 1 + sum(isinstance(x, _BRANCH) for x in ast.walk(n)),
                        "documented": ast.get_docstring(n) is not None})
                elif isinstance(n, ast.ClassDef):
                    entry["classes"].append({"name": n.name, "line": n.lineno})
            entry["risk"] = [msg for _, msg in self.scanner.scan(src, rel)]
            files.append(entry)

        funcs = [(f["path"], fn) for f in files for fn in f.get("functions", [])]
        return {
            "root": str(self.root),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "totals": {"files": len(files), "loc": sum(f.get("loc", 0) for f in files),
                       "functions": len(funcs),
                       "classes": sum(len(f.get("classes", [])) for f in files)},
            "hotspots": sorted(
                ({"path": p, "function": fn["name"], "line": fn["line"], "complexity": fn["complexity"]}
                 for p, fn in funcs if fn["complexity"] >= 10),
                key=lambda d: -d["complexity"])[:10],
            "undocumented": [f"{p}:{fn['name']}" for p, fn in funcs if not fn["documented"]][:25],
            "modules_without_tests": [f["path"] for f in files
                                      if Path(f["path"]).stem not in test_stems
                                      and not Path(f["path"]).name.startswith(("test_", "__"))][:25],
            "risk_findings": [r for f in files for r in f.get("risk", [])][:50],
            "broken_files": [f["path"] for f in files if "error" in f],
            "files": files,
        }


# --------------------------------------------------------------------------- #
# Proposal model + providers
# --------------------------------------------------------------------------- #


@dataclass
class FileChange:
    path: str
    action: str  # "modify" | "create"
    edits: list = field(default_factory=list)
    content: Optional[str] = None


@dataclass
class PatchProposal:
    title: str
    rationale: str
    changes: list
    id: str = field(default_factory=lambda: time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6])

    @classmethod
    def from_dict(cls, d: dict) -> "PatchProposal":
        if not isinstance(d, dict) or not isinstance(d.get("changes"), list) or not d["changes"]:
            raise ValidationError("proposal needs a non-empty 'changes' list")
        changes = []
        for c in d["changes"]:
            action = c.get("action")
            if action == "modify":
                edits = c.get("edits")
                if not isinstance(edits, list) or not edits or not all(
                        isinstance(e, dict) and isinstance(e.get("search"), str)
                        and isinstance(e.get("replace"), str) and e["search"] for e in edits):
                    raise ValidationError(f"bad edits for {c.get('path')}")
                changes.append(FileChange(c["path"], "modify", edits=edits))
            elif action == "create":
                if not isinstance(c.get("content"), str):
                    raise ValidationError(f"'content' required for {c.get('path')}")
                changes.append(FileChange(c["path"], "create", content=c["content"]))
            else:
                raise ValidationError(f"unsupported action {action!r} (delete is never allowed)")
        return cls(title=str(d.get("title", "untitled"))[:200],
                   rationale=str(d.get("rationale", ""))[:2000], changes=changes)


class PatchProvider(Protocol):
    def propose(self, goal: str, analysis: dict) -> PatchProposal: ...


class JsonFileProvider:
    def __init__(self, path: Path):
        self.path = Path(path)

    def propose(self, goal: str = "", analysis: Optional[dict] = None) -> PatchProposal:
        return PatchProposal.from_dict(json.loads(self.path.read_text(encoding="utf-8")))


class LocalCommandProvider:
    """Pipes a prompt to a LOCAL model command (e.g. `ollama run codellama`)."""

    def __init__(self, command: list, timeout: int = 600):
        self.command, self.timeout = command, timeout

    @staticmethod
    def _prompt(goal: str, analysis: dict) -> str:
        brief = {"totals": analysis["totals"], "hotspots": analysis["hotspots"],
                 "files": [{"path": f["path"], "functions": [x["name"] for x in f.get("functions", [])][:15]}
                           for f in analysis["files"][:60]]}
        return (
            "You are a careful software engineer. Respond with ONE JSON object only, matching:\n"
            '{"title": str, "rationale": str, "changes": [{"path": str, "action": "modify"|"create", '
            '"edits": [{"search": str, "replace": str}], "content": str}]}\n'
            "Rules: 'search' must match the existing file exactly once; use only relative paths; "
            "no network, subprocess, eval/exec, or file deletion.\n"
            f"GOAL: {goal}\nCODEBASE SUMMARY: {json.dumps(brief)}\n")

    def propose(self, goal: str, analysis: dict) -> PatchProposal:
        r = subprocess.run(self.command, input=self._prompt(goal, analysis), capture_output=True,
                           text=True, timeout=self.timeout, shell=False)
        if r.returncode != 0:
            raise ExecutorError(f"model command failed: {r.stderr[-500:]}")
        a, b = r.stdout.find("{"), r.stdout.rfind("}")
        if a < 0 or b <= a:
            raise ValidationError("model output contained no JSON object")
        return PatchProposal.from_dict(json.loads(r.stdout[a:b + 1]))


# --------------------------------------------------------------------------- #
# Sandbox
# --------------------------------------------------------------------------- #


@functools.lru_cache(maxsize=1)
def _netns_prefix() -> tuple:
    exe = shutil.which("unshare")
    if exe and sys.platform.startswith("linux"):
        try:
            if subprocess.run([exe, "-rn", "true"], capture_output=True, timeout=5).returncode == 0:
                return (exe, "-rn")
        except Exception:
            pass
    return ()


class Sandbox:
    """Runs checks against a throw-away copy of the project."""

    def __init__(self, cfg: ExecutorConfig):
        self.cfg = cfg
        self.tmp: Optional[Path] = None

    def __enter__(self) -> "Sandbox":
        self.tmp = Path(tempfile.mkdtemp(prefix="zentrax-sbx-"))
        self.project = self.tmp / "project"
        shutil.copytree(self.cfg.root, self.project, symlinks=True,
                        ignore=shutil.ignore_patterns(*self.cfg.ignore_dirs))
        return self

    def __exit__(self, *exc) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _limits(self):
        mem, cpu = self.cfg.memory_limit_mb * 1024 * 1024, self.cfg.test_timeout

        def apply():
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
            resource.setrlimit(resource.RLIMIT_FSIZE, (50 * 2**20, 50 * 2**20))
        return apply if os.name == "posix" else None

    def _run(self, cmd: list) -> tuple:
        prefix = _netns_prefix()
        if self.cfg.require_isolation and not prefix:
            raise SandboxFailure("network isolation unavailable and --require-isolation is set")
        env = {"PATH": os.environ.get("PATH", ""), "HOME": str(self.tmp), "LANG": "C.UTF-8",
               "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0", "ZENTRAX_SANDBOX": "1"}
        proc = subprocess.Popen([*prefix, *cmd], cwd=self.project, env=env, shell=False,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                preexec_fn=self._limits(), start_new_session=(os.name == "posix"))
        try:
            out, _ = proc.communicate(timeout=self.cfg.test_timeout)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(Exception):
                os.killpg(proc.pid, signal.SIGKILL) if os.name == "posix" else proc.kill()
            proc.communicate()
            raise SandboxFailure(f"test run exceeded {self.cfg.test_timeout}s")
        return proc.returncode, out

    def verify(self, prepared: list) -> list:
        notes = []
        for p in prepared:
            dest = self.project / p.rel
            _atomic_write(dest, p.after)
            if dest.suffix == ".py":
                try:
                    py_compile.compile(str(dest), cfile=str(self.tmp / "x.pyc"), doraise=True)
                except py_compile.PyCompileError as e:
                    raise SandboxFailure(f"compile failed: {e.msg}")
        cmd = self.cfg.test_command
        if cmd is None and ((self.project / "tests").is_dir() or (self.project / "pytest.ini").exists()):
            cmd = (sys.executable, "-m", "pytest", "-x", "-q")
        if cmd is None:
            return notes + ["no test suite detected; only compile checks ran"]
        rc, out = self._run(list(cmd))
        if rc == 5:
            notes.append("pytest collected no tests")
        elif rc != 0:
            raise SandboxFailure(f"tests failed (exit {rc}):\n{out[-2000:]}")
        else:
            notes.append("tests passed in sandbox")
        if not _netns_prefix():
            notes.append("WARNING: no network namespace available; test run was not network-isolated")
        return notes


# --------------------------------------------------------------------------- #
# State: backups, lock, tamper-evident audit log
# --------------------------------------------------------------------------- #


class StateStore:
    def __init__(self, cfg: ExecutorConfig):
        self.dir = cfg.state_dir
        self.backups = self.dir / "backups"
        self.log_path = self.dir / "audit.jsonl"
        self.backups.mkdir(parents=True, exist_ok=True, mode=0o700)

    @contextlib.contextmanager
    def lock(self):
        path = self.dir / "executor.lock"
        if path.exists():
            try:
                os.kill(int(path.read_text() or 0), 0)
                raise ExecutorError("another executor run is active")
            except (ProcessLookupError, ValueError):
                path.unlink(missing_ok=True)  # stale lock
            except PermissionError:
                raise ExecutorError("another executor run is active")
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            os.write(fd, str(os.getpid()).encode())
            yield
        finally:
            os.close(fd)
            path.unlink(missing_ok=True)

    def _last_hash(self) -> str:
        if not self.log_path.exists():
            return "0" * 64
        lines = self.log_path.read_text(encoding="utf-8").strip().splitlines()
        return json.loads(lines[-1])["hash"] if lines else "0" * 64

    def audit(self, event: str, **data) -> None:
        entry = {"ts": time.time(), "event": event, "data": data, "prev": self._last_hash()}
        entry["hash"] = _sha(json.dumps(entry, sort_keys=True).encode())
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")

    def verify_log(self) -> bool:
        prev = "0" * 64
        if not self.log_path.exists():
            return True
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            h = e.pop("hash")
            if e["prev"] != prev or _sha(json.dumps(e, sort_keys=True).encode()) != h:
                return False
            prev = h
        return True


# --------------------------------------------------------------------------- #
# Executor
# --------------------------------------------------------------------------- #


@dataclass
class Prepared:
    rel: str
    target: Path
    existed: bool
    before: Optional[str]
    after: str


@dataclass
class Report:
    status: str  # applied | dry_run_ok | rejected | declined | failed
    patch_id: str
    notes: list = field(default_factory=list)
    diff: str = ""


Approver = Callable[[PatchProposal, str], bool]


def interactive_approver(proposal: PatchProposal, diff: str) -> bool:
    if not sys.stdin.isatty():
        return False
    print(f"\n=== {proposal.title} ({proposal.id}) ===\n{proposal.rationale}\n\n{diff}")
    return input("Apply this patch? [y/N] ").strip().lower() == "y"


class SelfCodingExecutor:
    def __init__(self, cfg: ExecutorConfig):
        self.cfg = cfg
        self.guard = PathGuard(cfg)
        self.scanner = SafetyScanner(cfg)
        self.analyzer = CodebaseAnalyzer(cfg, self.scanner)
        self.store = StateStore(cfg)

    # -- stage 1
    def analyze(self) -> dict:
        report = self.analyzer.analyze()
        self.store.audit("analyze", totals=report["totals"])
        return report

    # -- stage 3
    @staticmethod
    def _apply_edits(rel: str, text: str, edits: list) -> str:
        for e in edits:
            if text.count(e["search"]) != 1:
                raise ValidationError(f"{rel}: 'search' text must match exactly once "
                                      f"(found {text.count(e['search'])})")
            text = text.replace(e["search"], e["replace"], 1)
        return text

    def prepare(self, proposal: PatchProposal) -> list:
        if len(proposal.changes) > self.cfg.max_files_per_patch:
            raise ValidationError(f"too many files (max {self.cfg.max_files_per_patch})")
        seen, prepared = set(), []
        for ch in proposal.changes:
            target, rel = self.guard.check_editable(ch.path)
            if rel in seen:
                raise ValidationError(f"duplicate change for {rel}")
            seen.add(rel)
            if ch.action == "modify":
                if not target.is_file():
                    raise ValidationError(f"{rel}: file does not exist")
                before = target.read_text(encoding="utf-8")
                after = self._apply_edits(rel, before, ch.edits)
            else:
                if target.exists():
                    raise ValidationError(f"{rel}: already exists (use 'modify')")
                before, after = None, ch.content
            if len(after.encode("utf-8")) > self.cfg.max_file_bytes:
                raise ValidationError(f"{rel}: exceeds {self.cfg.max_file_bytes} bytes")
            if after == before:
                raise ValidationError(f"{rel}: change is a no-op")
            if rel.endswith(".py"):
                new = self.scanner.scan(after, rel)
                old_keys = {k for k, _ in self.scanner.scan(before, rel)} if before else set()
                bad = [m for k, m in new if k == "syntax" or k not in old_keys]
                if bad:
                    raise SecurityViolation("; ".join(bad))
            prepared.append(Prepared(rel, target, target.exists(), before, after))
        return prepared

    @staticmethod
    def _diff(prepared: list) -> str:
        chunks = []
        for p in prepared:
            chunks.extend(difflib.unified_diff(
                (p.before or "").splitlines(keepends=True), p.after.splitlines(keepends=True),
                f"a/{p.rel}" if p.existed else "/dev/null", f"b/{p.rel}"))
        return "".join(chunks)

    # -- stages 3-7
    def execute(self, proposal: PatchProposal, *, dry_run: bool = False,
                approve: Optional[Approver] = None) -> Report:
        with self.store.lock():
            try:
                prepared = self.prepare(proposal)
                diff = self._diff(prepared)
                with Sandbox(self.cfg) as sbx:
                    notes = sbx.verify(prepared)
                if dry_run:
                    self.store.audit("dry_run_ok", id=proposal.id, title=proposal.title)
                    return Report("dry_run_ok", proposal.id, notes, diff)
                if not (approve or interactive_approver)(proposal, diff):
                    self.store.audit("declined", id=proposal.id)
                    return Report("declined", proposal.id, notes, diff)
                self._commit(proposal, prepared)
                notes.append(f"applied; rollback with: rollback {proposal.id}")
                return Report("applied", proposal.id, notes, diff)
            except (SecurityViolation, ValidationError) as e:
                self.store.audit("rejected", id=proposal.id, reason=str(e)[:500])
                return Report("rejected", proposal.id, [str(e)])
            except ExecutorError as e:
                self.store.audit("failed", id=proposal.id, reason=str(e)[:500])
                return Report("failed", proposal.id, [str(e)])

    # -- stage 6
    def _commit(self, proposal: PatchProposal, prepared: list) -> None:
        bdir = self.store.backups / proposal.id
        manifest = []
        for p in prepared:
            if p.existed:
                dest = bdir / "files" / p.rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p.target, dest)
            manifest.append({"path": p.rel, "existed": p.existed,
                             "sha_after": _sha(p.after.encode("utf-8"))})
        (bdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self.store.audit("commit_start", id=proposal.id, title=proposal.title,
                         files=[m["path"] for m in manifest])
        try:
            for p in prepared:
                _atomic_write(p.target, p.after)
            for p, m in zip(prepared, manifest):
                if _sha(p.target.read_bytes()) != m["sha_after"]:
                    raise ExecutorError(f"hash mismatch after writing {p.rel}")
                if p.rel.endswith(".py"):
                    ast.parse(p.target.read_text(encoding="utf-8"))
        except Exception as e:
            self.rollback(proposal.id, force=True)
            raise ExecutorError(f"commit failed and was rolled back: {e}")
        self.store.audit("commit_ok", id=proposal.id)

    # -- undo
    def rollback(self, patch_id: str, force: bool = False) -> list:
        if not patch_id.replace("-", "").isalnum():
            raise SecurityViolation("invalid patch id")
        bdir = self.store.backups / patch_id
        mf = bdir / "manifest.json"
        if not mf.is_file():
            raise ExecutorError(f"no backup for {patch_id}")
        restored = []
        for m in json.loads(mf.read_text(encoding="utf-8")):
            target = self.guard.resolve(m["path"])
            if target.exists() and _sha(target.read_bytes()) != m["sha_after"] and not force:
                raise ExecutorError(f"{m['path']} changed since the patch; use --force to override")
            if m["existed"]:
                shutil.copy2(bdir / "files" / m["path"], target)
            else:
                target.unlink(missing_ok=True)  # only files this patch created
            restored.append(m["path"])
        self.store.audit("rollback", id=patch_id, files=restored)
        return restored

    def history(self) -> list:
        if not self.store.log_path.exists():
            return []
        return [json.loads(l) for l in self.store.log_path.read_text(encoding="utf-8").splitlines()]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _print_report(r: Report) -> int:
    print(f"[{r.status.upper()}] {r.patch_id}")
    for n in r.notes:
        print(f"  - {n}")
    if r.diff and r.status in ("dry_run_ok", "applied"):
        print("\n" + r.diff)
    return 0 if r.status in ("dry_run_ok", "applied") else 1


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="ZentraX-AI self-coding executor (local, gated)")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--test-cmd", help="override the sandbox test command")
    ap.add_argument("--require-isolation", action="store_true",
                    help="fail closed if network isolation is unavailable")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze"); a.add_argument("--json", action="store_true")
    sub.add_parser("check").add_argument("patch", type=Path)
    p = sub.add_parser("apply"); p.add_argument("patch", type=Path); p.add_argument("--yes", action="store_true")
    g = sub.add_parser("generate")
    g.add_argument("--goal", required=True); g.add_argument("--model-cmd", required=True)
    g.add_argument("--out", type=Path, default=Path("patch.json"))
    r = sub.add_parser("rollback"); r.add_argument("patch_id"); r.add_argument("--force", action="store_true")
    sub.add_parser("history")
    sub.add_parser("verify-log")
    args = ap.parse_args(argv)

    cfg = ExecutorConfig(
        root=args.root.resolve(),
        test_command=tuple(shlex.split(args.test_cmd)) if args.test_cmd else None,
        require_isolation=args.require_isolation)
    ex = SelfCodingExecutor(cfg)

    try:
        if args.cmd == "analyze":
            rep = ex.analyze()
            if args.json:
                print(json.dumps(rep, indent=2))
            else:
                rep.pop("files")
                print(json.dumps(rep, indent=2))
            return 0
        if args.cmd == "check":
            return _print_report(ex.execute(JsonFileProvider(args.patch).propose(""), dry_run=True))
        if args.cmd == "apply":
            prop = JsonFileProvider(args.patch).propose("")
            return _print_report(ex.execute(prop, approve=(lambda *_: True) if args.yes else None))
        if args.cmd == "generate":
            prop = LocalCommandProvider(shlex.split(args.model_cmd)).propose(args.goal, ex.analyze())
            args.out.write_text(json.dumps({
                "title": prop.title, "rationale": prop.rationale,
                "changes": [{k: v for k, v in vars(c).items() if v not in (None, [])} for c in prop.changes]},
                indent=2), encoding="utf-8")
            print(f"Proposal written to {args.out}. Review it, then run: check / apply")
            return 0
        if args.cmd == "rollback":
            print("restored:", *ex.rollback(args.patch_id, args.force))
            return 0
        if args.cmd == "history":
            for e in ex.history():
                print(time.strftime("%F %T", time.localtime(e["ts"])), e["event"], json.dumps(e["data"]))
            return 0
        if args.cmd == "verify-log":
            ok = ex.store.verify_log()
            print("audit log intact" if ok else "AUDIT LOG TAMPERED")
            return 0 if ok else 2
    except ExecutorError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
