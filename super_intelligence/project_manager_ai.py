"""
ZentraX-AI :: super_intelligence/project_manager_ai.py

Autonomous, offline-only project manager.

Every cycle the manager:
  1. Honors the kill switch and verifies its own database integrity.
  2. Collects findings from the repository: syntax errors, test gaps, TODO/FIXME
     markers, missing project files, hardcoded secrets, file permissions, git
     activity, and open recommendations from core_growth_engine.
  3. Turns findings into tasks and ranks them with a deterministic score.
  4. Auto-resolves tasks whose underlying issue has disappeared.
  5. Decides the current growth-pipeline stage (secure -> stabilize ->
     foundation -> expand) and selects a focus list for that stage.
  6. Publishes data/PROJECT_PLAN.md and a 0-100 repository health score.
  7. Tunes its own schedule within hard bounds and logs everything to a
     tamper-evident, hash-chained audit log.

Safety design (deliberate):
  * Standard library only: no sockets, no HTTP, no subprocess, no eval/exec.
  * Never modifies project source. It writes only to its own SQLite database
    and to data/PROJECT_PLAN.md.
  * Secret findings record file, line and type only, never the secret value.
  * Reads the growth engine's database in read-only mode.
  * The KILL_SWITCH file halts all activity immediately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

try:  # package import (python -m super_intelligence.project_manager_ai)
    from .core_growth_engine import EXCLUDED_DIRS, EngineConfig, SecureStore, scan_project
except ImportError:  # direct script execution
    from core_growth_engine import EXCLUDED_DIRS, EngineConfig, SecureStore, scan_project  # type: ignore

log = logging.getLogger("zentrax.project_manager")

STAGES = ("secure", "stabilize", "foundation", "expand")
STAGE_SOURCES = {
    "secure": {"security"},
    "stabilize": {"syntax", "growth_engine"},
    "foundation": {"tests", "structure", "git"},
    "expand": {"todo", "growth_engine", "git"},
}
SEVERITY_BASE = {"high": 80, "medium": 50, "low": 20}
SOURCE_BONUS = {"security": 15, "syntax": 10}
SECRET_SCAN_EXTS = {".py", ".json", ".yml", ".yaml", ".toml", ".ini", ".cfg", ".env"}

SECRET_PATTERNS = [
    ("private key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("hardcoded credential", re.compile(
        r"""(?i)\b(?:api[_-]?key|secret|token|passwd|password)\b\s*[:=]\s*['"]([^'"\s]{8,})['"]""")),
]
PLACEHOLDER = re.compile(r"(?i)your|example|changeme|xxx|<|\$\{|dummy|test|placeholder|\*{3,}")
TODO_RE = re.compile(r"#\s*(TODO|FIXME|HACK|XXX)\b:?\s*(.*)")


# --------------------------------------------------------------------------- #
# Configuration & data types
# --------------------------------------------------------------------------- #
@dataclass
class ManagerConfig:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    db_path: Optional[Path] = None
    growth_db_path: Optional[Path] = None
    report_path: Optional[Path] = None
    kill_switch_path: Optional[Path] = None
    min_interval: float = 60.0
    max_interval: float = 3600.0
    initial_interval: float = 300.0
    focus_size: int = 5
    retention_days: int = 90
    stale_commit_days: int = 14
    max_todo_tasks: int = 200
    max_files: int = 3000
    max_file_bytes: int = 500_000

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root).resolve()
        data = self.project_root / "data"
        self.db_path = Path(self.db_path or data / "project_manager.db").resolve()
        self.growth_db_path = Path(self.growth_db_path or data / "growth_engine.db").resolve()
        self.report_path = Path(self.report_path or data / "PROJECT_PLAN.md").resolve()
        self.kill_switch_path = Path(self.kill_switch_path or self.project_root / "KILL_SWITCH").resolve()


@dataclass(frozen=True)
class Finding:
    key: str
    title: str
    source: str
    severity: str   # high | medium | low
    effort: int     # 1 (minutes) .. 5 (days)
    location: str = ""


# --------------------------------------------------------------------------- #
# Secure task store (extends the growth engine's hardened SecureStore)
# --------------------------------------------------------------------------- #
class ManagerStore(SecureStore):
    SCHEMA = SecureStore.SCHEMA + """
    CREATE TABLE IF NOT EXISTS tasks (
        key TEXT PRIMARY KEY, title TEXT NOT NULL, source TEXT NOT NULL, severity TEXT NOT NULL,
        effort INTEGER NOT NULL, location TEXT NOT NULL DEFAULT '', score REAL NOT NULL DEFAULT 0,
        status TEXT NOT NULL DEFAULT 'open', first_seen REAL NOT NULL, last_seen REAL NOT NULL,
        resolved_at REAL);
    CREATE INDEX IF NOT EXISTS idx_tasks_status_score ON tasks(status, score);
    CREATE TABLE IF NOT EXISTS pipeline_history (
        id INTEGER PRIMARY KEY, ts REAL NOT NULL, stage TEXT NOT NULL, reason TEXT NOT NULL);
    """

    def upsert_task(self, f: Finding) -> None:
        now = time.time()
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO tasks(key,title,source,severity,effort,location,score,status,first_seen,last_seen)
                   VALUES (?,?,?,?,?,?,0,'open',?,?)
                   ON CONFLICT(key) DO UPDATE SET
                     title=excluded.title, severity=excluded.severity, effort=excluded.effort,
                     location=excluded.location, last_seen=excluded.last_seen,
                     status=CASE WHEN tasks.status='resolved' THEN 'open' ELSE tasks.status END,
                     resolved_at=CASE WHEN tasks.status='resolved' THEN NULL ELSE tasks.resolved_at END""",
                (f.key, f.title, f.source, f.severity, f.effort, f.location, now, now),
            )

    def resolve_missing(self, seen: set[str], ok_sources: set[str]) -> int:
        """Auto-resolve open tasks whose source ran successfully but no longer reports them."""
        resolved = 0
        with self._lock, self._conn:
            rows = self._conn.execute("SELECT key, source FROM tasks WHERE status='open'").fetchall()
            for r in rows:
                if r["source"] in ok_sources and r["key"] not in seen:
                    self._conn.execute(
                        "UPDATE tasks SET status='resolved', resolved_at=? WHERE key=?", (time.time(), r["key"]))
                    resolved += 1
        return resolved

    def rescore(self) -> None:
        now = time.time()
        with self._lock, self._conn:
            for r in self._conn.execute(
                    "SELECT key, source, severity, effort, first_seen FROM tasks WHERE status='open'").fetchall():
                age_days = min((now - r["first_seen"]) / 86400, 30)
                score = (SEVERITY_BASE.get(r["severity"], 20) + SOURCE_BONUS.get(r["source"], 0)
                         + age_days * 0.5 - r["effort"] * 2)
                self._conn.execute("UPDATE tasks SET score=? WHERE key=?", (round(score, 2), r["key"]))

    def open_tasks(self, limit: Optional[int] = None) -> list[dict[str, Any]]:
        sql = ("SELECT key,title,source,severity,effort,location,score,first_seen FROM tasks "
               "WHERE status='open' ORDER BY score DESC, key")
        with self._lock:
            rows = self._conn.execute(sql + (" LIMIT ?" if limit else ""), (limit,) if limit else ()).fetchall()
        return [dict(r) for r in rows]

    def dismiss(self, key: str) -> bool:
        with self._lock, self._conn:
            return self._conn.execute(
                "UPDATE tasks SET status='dismissed' WHERE key=? AND status='open'", (key,)).rowcount > 0

    def prune_tasks(self, days: int) -> int:
        cutoff = time.time() - days * 86400
        with self._lock, self._conn:
            return self._conn.execute(
                "DELETE FROM tasks WHERE status IN ('resolved','dismissed') AND COALESCE(resolved_at,last_seen)<?",
                (cutoff,)).rowcount

    def set_stage(self, stage: str, reason: str) -> bool:
        with self._lock, self._conn:
            row = self._conn.execute("SELECT stage FROM pipeline_history ORDER BY id DESC LIMIT 1").fetchone()
            if row and row["stage"] == stage:
                return False
            self._conn.execute(
                "INSERT INTO pipeline_history(ts, stage, reason) VALUES (?,?,?)", (time.time(), stage, reason))
            return True


# --------------------------------------------------------------------------- #
# Collectors (all read-only)
# --------------------------------------------------------------------------- #
def iter_files(root: Path, exts: set[str], max_files: int, max_bytes: int) -> Iterator[Path]:
    count = 0
    for dirpath, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED_DIRS)
        for name in sorted(names):
            p = Path(dirpath) / name
            if p.suffix.lower() not in exts and name not in exts:
                continue
            try:
                if p.is_symlink() or p.stat().st_size > max_bytes:
                    continue
            except OSError:
                continue
            count += 1
            if count > max_files:
                return
            yield p


def _read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _sha(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def collect_todos(cfg: ManagerConfig) -> list[Finding]:
    out: list[Finding] = []
    for p in iter_files(cfg.project_root, {".py"}, cfg.max_files, cfg.max_file_bytes):
        rel = str(p.relative_to(cfg.project_root))
        for no, line in enumerate(_read(p).splitlines(), 1):
            m = TODO_RE.search(line)
            if not m:
                continue
            tag, text = m.group(1), m.group(2).strip()[:100]
            sev = "medium" if tag in ("FIXME", "HACK") else "low"
            out.append(Finding(f"todo:{_sha(rel + tag + text)}", f"{tag}: {text or '(no description)'}",
                               "todo", sev, 2, f"{rel}:{no}"))
            if len(out) >= cfg.max_todo_tasks:
                return out
    return out


def collect_structure(cfg: ManagerConfig) -> list[Finding]:
    root, out = cfg.project_root, []
    if not any((root / n).exists() for n in ("README.md", "README.rst", "README")):
        out.append(Finding("structure:readme", "Add a README describing setup and architecture", "structure", "medium", 1))
    if not any((root / n).exists() for n in ("requirements.txt", "pyproject.toml", "Pipfile")):
        out.append(Finding("structure:deps", "Add a dependency manifest (requirements.txt or pyproject.toml)",
                           "structure", "medium", 1))
    if not (root / ".gitignore").exists():
        out.append(Finding("structure:gitignore", "Add .gitignore (exclude data/, .env, virtualenvs)",
                           "structure", "medium", 1))
    if not any((root / n).exists() for n in ("LICENSE", "LICENSE.md", "LICENSE.txt")):
        out.append(Finding("structure:license", "Add a LICENSE file", "structure", "low", 1))
    return out


def collect_scan(cfg: ManagerConfig) -> list[Finding]:
    scan = scan_project(EngineConfig(project_root=cfg.project_root))
    out = [Finding(f"syntax:{_sha(e)}", f"File does not parse: {e}", "syntax", "high", 2, e.split(":")[0])
           for e in scan.syntax_errors]
    m = scan.metrics
    if m["py_files"] and m["test_files"] / m["py_files"] < 0.15:
        sev = "high" if m["test_files"] == 0 and m["py_files"] >= 10 else "medium"
        out.append(Finding("tests:coverage", "Test files are under 15% of Python files; add tests for core modules",
                           "tests", sev, 5))
    return out


def collect_security(cfg: ManagerConfig) -> list[Finding]:
    out: list[Finding] = []
    for p in iter_files(cfg.project_root, SECRET_SCAN_EXTS, cfg.max_files, cfg.max_file_bytes):
        rel = str(p.relative_to(cfg.project_root))
        for no, line in enumerate(_read(p).splitlines(), 1):
            for kind, rx in SECRET_PATTERNS:
                m = rx.search(line)
                if not m:
                    continue
                if kind == "hardcoded credential" and PLACEHOLDER.search(m.group(1)):
                    continue
                # value is deliberately never stored
                out.append(Finding(f"sec:{_sha(f'{rel}|{kind}|{line.strip()}')}",
                                   f"Possible {kind} committed in source; move it to local env or secret store",
                                   "security", "high", 1, f"{rel}:{no}"))
    if os.name == "posix":
        for db in (cfg.db_path, cfg.growth_db_path):
            try:
                if db.exists() and db.stat().st_mode & 0o077:
                    out.append(Finding(f"sec:perm:{db.name}", f"Database {db.name} is accessible by other users (expected 0600)",
                                       "security", "medium", 1, str(db.relative_to(cfg.project_root))
                                       if db.is_relative_to(cfg.project_root) else str(db)))
            except OSError:
                pass
    return out


def collect_git(cfg: ManagerConfig) -> list[Finding]:
    git = cfg.project_root / ".git"
    if not git.exists():
        return [Finding("git:missing", "Project is not under version control", "git", "medium", 1)]
    try:
        last = (git / "logs" / "HEAD").read_text(errors="replace").strip().splitlines()[-1]
        ts = int(last.split("\t")[0].split()[-2])
    except (OSError, IndexError, ValueError):
        return []
    age = (time.time() - ts) / 86400
    if age > cfg.stale_commit_days:
        return [Finding("git:stale", f"No commits in {int(age)} days; commit or review pending work", "git", "low", 1)]
    return []


def collect_growth_recs(cfg: ManagerConfig) -> list[Finding]:
    if not cfg.growth_db_path.exists():
        return []
    skip = {"syntax_errors", "test_coverage"}  # already covered by collect_scan
    sev_map = {"high": "high", "medium": "medium", "low": "low"}
    try:
        conn = sqlite3.connect(cfg.growth_db_path.as_uri() + "?mode=ro", uri=True, timeout=5)
        try:
            rows = conn.execute("SELECT key, severity, message FROM recommendations WHERE status='open'").fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return []
    return [Finding(f"ge:{k}", msg[:160], "growth_engine", sev_map.get(sev, "low"), 3)
            for k, sev, msg in rows if k not in skip]


# --------------------------------------------------------------------------- #
# Pipeline logic
# --------------------------------------------------------------------------- #
def decide_stage(tasks: list[dict[str, Any]]) -> tuple[str, str]:
    def has(pred: Callable[[dict[str, Any]], bool]) -> bool:
        return any(pred(t) for t in tasks)

    if has(lambda t: t["source"] == "security" and t["severity"] == "high"):
        return "secure", "High-severity security finding is open"
    if has(lambda t: t["severity"] == "high"):
        return "stabilize", "High-severity defect is open"
    if has(lambda t: t["source"] == "security"):
        return "secure", "Security findings remain"
    if has(lambda t: t["source"] in STAGE_SOURCES["foundation"] and t["severity"] != "low"):
        return "foundation", "Tests, structure or version-control gaps remain"
    return "expand", "Repository is healthy; work the improvement backlog"


def health_score(tasks: list[dict[str, Any]]) -> int:
    penalty = sum({"high": 25, "medium": 8, "low": 2}.get(t["severity"], 2) for t in tasks)
    return max(0, 100 - penalty)


# --------------------------------------------------------------------------- #
# Manager
# --------------------------------------------------------------------------- #
class ProjectManagerAI:
    """Autonomous offline project manager. Thread-based; start it from a FastAPI lifespan."""

    def __init__(self, config: Optional[ManagerConfig] = None) -> None:
        self.cfg = config or ManagerConfig()
        self.store = ManagerStore(self.cfg.db_path)  # type: ignore[arg-type]
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cycles = 0
        self._failures = 0
        self._quiet = 0
        self._last_fp: Optional[str] = None
        self._last_report: dict[str, Any] = {}
        self.interval = self._clamp(self.store.get_tunable("pm_interval", self.cfg.initial_interval))
        self.store.audit("pm_init", {"root": str(self.cfg.project_root), "interval": self.interval})

    # -- lifecycle --------------------------------------------------------- #
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="zentrax-project-manager", daemon=True)
        self._thread.start()
        log.info("Project manager started (interval=%.0fs)", self.interval)

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)
        self.store.audit("pm_stop", {"cycles": self._cycles})
        self.store.close()

    def kill_switch_active(self) -> bool:
        return self.cfg.kill_switch_path.exists()  # type: ignore[union-attr]

    def _clamp(self, v: float) -> float:
        return max(self.cfg.min_interval, min(self.cfg.max_interval, v))

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_cycle()
                self._failures = 0
            except Exception as exc:  # never crash the host application
                self._failures += 1
                log.exception("Project manager cycle failed")
                try:
                    self.store.audit("pm_cycle_error", {"error": repr(exc), "failures": self._failures})
                except Exception:
                    pass
            backoff = min(2 ** self._failures, 32) if self._failures else 1
            self._stop.wait(self.interval * backoff)

    # -- one full cycle ---------------------------------------------------- #
    def run_cycle(self) -> dict[str, Any]:
        if self.kill_switch_active():
            self.store.audit("pm_kill_switch_active")
            self._last_report = {"status": "halted_by_kill_switch"}
            return self._last_report
        if not self.store.integrity_ok():
            self.store.audit("pm_db_integrity_failure")
            self._last_report = {"status": "halted_db_integrity_failure"}
            log.error("Project manager database failed integrity check; paused.")
            return self._last_report

        audit_ok = self.store.verify_audit()
        collectors: list[tuple[str, Callable[[ManagerConfig], list[Finding]], set[str]]] = [
            ("todos", collect_todos, {"todo"}),
            ("structure", collect_structure, {"structure"}),
            ("scan", collect_scan, {"syntax", "tests"}),
            ("security", collect_security, {"security"}),
            ("git", collect_git, {"git"}),
            ("growth_engine", collect_growth_recs, {"growth_engine"}),
        ]
        findings: list[Finding] = []
        ok_sources: set[str] = set()
        failed: list[str] = []
        for name, fn, sources in collectors:
            try:
                findings.extend(fn(self.cfg))
                ok_sources |= sources   # only auto-resolve tasks for collectors that actually ran
            except Exception as exc:
                failed.append(name)
                log.warning("Collector %s failed: %r", name, exc)
        if not audit_ok:
            findings.append(Finding("sec:audit_chain", "Manager audit log hash chain is broken; the local database may be tampered with",
                                    "security", "high", 2))
            ok_sources.add("security")

        for f in findings:
            self.store.upsert_task(f)
        resolved = self.store.resolve_missing({f.key for f in findings}, ok_sources)
        self.store.rescore()
        self.store.prune_tasks(self.cfg.retention_days)

        tasks = self.store.open_tasks()
        stage, reason = decide_stage(tasks)
        stage_changed = self.store.set_stage(stage, reason)
        focus = [t for t in tasks if t["source"] in STAGE_SOURCES[stage]][: self.cfg.focus_size] \
            or tasks[: self.cfg.focus_size]
        health = health_score(tasks)

        self._write_report(stage, reason, health, focus, tasks)
        tuning = self._self_tune(tasks)
        maintenance = self.store.maintain(self.cfg.retention_days, 0.20)

        self._cycles += 1
        self._last_report = {
            "status": "ok", "cycle": self._cycles, "health": health, "stage": stage, "stage_reason": reason,
            "stage_changed": stage_changed, "open_tasks": len(tasks), "auto_resolved": resolved,
            "failed_collectors": failed, "focus": focus, "tuning": tuning, "audit_chain_valid": audit_ok,
        }
        self.store.audit("pm_cycle_complete", {
            "cycle": self._cycles, "health": health, "stage": stage, "open": len(tasks),
            "resolved": resolved, "failed": failed, "maintenance": maintenance})
        return self._last_report

    # -- self-tuning (bounded schedule only) ------------------------------- #
    def _self_tune(self, tasks: list[dict[str, Any]]) -> dict[str, Any]:
        fp = hashlib.sha256(json.dumps(sorted((t["key"], t["severity"]) for t in tasks)).encode()).hexdigest()
        changed, self._last_fp = fp != self._last_fp, fp
        old = self.interval
        if changed:
            self._quiet = 0
            self.interval = self._clamp(self.interval * 0.5)
        else:
            self._quiet += 1
            if self._quiet >= 3:
                self.interval = self._clamp(self.interval * 1.5)
        if self.interval != old:
            self.store.set_tunable("pm_interval", self.interval)
        return {"tasks_changed": changed, "interval": round(self.interval, 1)}

    # -- report ------------------------------------------------------------ #
    def _write_report(self, stage: str, reason: str, health: int,
                      focus: list[dict[str, Any]], tasks: list[dict[str, Any]]) -> None:
        counts = {s: sum(1 for t in tasks if t["severity"] == s) for s in ("high", "medium", "low")}
        lines = [
            "# ZentraX-AI Project Plan (auto-generated, do not edit)",
            "",
            f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"- Health: **{health}/100**",
            f"- Pipeline stage: **{stage}** ({reason})",
            f"- Open tasks: {len(tasks)} (high {counts['high']}, medium {counts['medium']}, low {counts['low']})",
            "",
            "## Focus list",
        ]
        if focus:
            for i, t in enumerate(focus, 1):
                loc = f" `{t['location']}`" if t["location"] else ""
                lines.append(f"{i}. [{t['severity'].upper()}] {t['title']}{loc} (score {t['score']}, effort {t['effort']}/5)")
        else:
            lines.append("Nothing open. Repository is clean.")
        lines += ["", "## Pipeline", " -> ".join(f"**{s}**" if s == stage else s for s in STAGES), ""]
        path: Path = self.cfg.report_path  # type: ignore[assignment]
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text("\n".join(lines), encoding="utf-8")
            try:
                os.chmod(tmp, 0o600)
            except OSError:
                pass
            os.replace(tmp, path)  # atomic
        except OSError as exc:
            log.warning("Could not write plan report: %r", exc)

    # -- public API (for FastAPI routes) ----------------------------------- #
    def status(self) -> dict[str, Any]:
        return {
            "running": bool(self._thread and self._thread.is_alive()),
            "kill_switch": self.kill_switch_active(),
            "interval_seconds": round(self.interval, 1),
            "cycles": self._cycles,
            "last_report": self._last_report,
        }

    def tasks(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.store.open_tasks(limit)

    def dismiss_task(self, key: str) -> bool:
        ok = self.store.dismiss(key)
        if ok:
            self.store.audit("pm_task_dismissed", {"key": key})
        return ok


# --------------------------------------------------------------------------- #
# Singleton helper + CLI
# --------------------------------------------------------------------------- #
_manager: Optional[ProjectManagerAI] = None
_manager_lock = threading.Lock()


def get_manager(config: Optional[ManagerConfig] = None) -> ProjectManagerAI:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = ProjectManagerAI(config)
        return _manager


def main() -> None:
    parser = argparse.ArgumentParser(description="ZentraX-AI autonomous project manager (offline)")
    parser.add_argument("--once", action="store_true", help="run one cycle and print the report")
    parser.add_argument("--root", type=Path, default=None, help="project root (defaults to repo root)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    mgr = ProjectManagerAI(ManagerConfig(project_root=args.root) if args.root else ManagerConfig())
    try:
        if args.once:
            print(json.dumps(mgr.run_cycle(), indent=2, default=str))
        else:
            mgr.start()
            while True:
                time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        mgr.stop()


if __name__ == "__main__":
    main()
