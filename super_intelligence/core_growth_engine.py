"""
ZentraX-AI :: super_intelligence/core_growth_engine.py

Autonomous, offline-only growth engine.

Every cycle the engine:
  1. Checks the kill switch and verifies database integrity.
  2. Scans the project (read-only, AST-based) and records metrics.
  3. Analyzes trends and anomalies in its own local history.
  4. Self-optimizes within hard bounds (schedule, DB maintenance, retention).
  5. Writes prioritized growth recommendations.
  6. Appends every action to a tamper-evident, hash-chained audit log.

Safety design (deliberate):
  * Standard library only. No sockets, no HTTP, no subprocess, no eval/exec.
  * Never modifies project source files. "Growth" means measuring, analyzing
    and recommending; the engine writes only to its own SQLite database.
  * Self-optimization tunes bounded parameters, never its own code.
  * Database file is owner-only (0600), uses parameterized SQL, and the audit
    log is hash-chained so tampering is detectable.
  * A kill switch file stops all activity immediately.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import os
import sqlite3
import statistics
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("zentrax.growth_engine")

EXCLUDED_DIRS = {
    ".git", ".venv", "venv", "env", "__pycache__", "node_modules",
    "build", "dist", ".mypy_cache", ".pytest_cache", "data",
}
NETWORK_MODULES = {"socket", "requests", "httpx", "urllib", "urllib3", "aiohttp", "http", "ftplib", "smtplib"}
GENESIS_HASH = "0" * 64


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class EngineConfig:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    db_path: Optional[Path] = None
    kill_switch_path: Optional[Path] = None
    min_interval: float = 30.0          # seconds, lower bound for self-tuned schedule
    max_interval: float = 3600.0        # seconds, upper bound
    initial_interval: float = 120.0
    retention_days: int = 90
    max_files: int = 5000
    max_file_bytes: int = 1_000_000
    fragmentation_vacuum_ratio: float = 0.20
    anomaly_zscore: float = 3.0
    history_window: int = 30

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root).resolve()
        if self.db_path is None:
            self.db_path = self.project_root / "data" / "growth_engine.db"
        if self.kill_switch_path is None:
            self.kill_switch_path = self.project_root / "KILL_SWITCH"
        self.db_path = Path(self.db_path).resolve()
        self.kill_switch_path = Path(self.kill_switch_path).resolve()


# --------------------------------------------------------------------------- #
# Secure local store
# --------------------------------------------------------------------------- #
class SecureStore:
    """Local SQLite store: owner-only permissions, parameterized SQL, hash-chained audit."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS metrics (
        id INTEGER PRIMARY KEY, ts REAL NOT NULL, name TEXT NOT NULL, value REAL NOT NULL);
    CREATE INDEX IF NOT EXISTS idx_metrics_name_ts ON metrics(name, ts);
    CREATE TABLE IF NOT EXISTS tunables (
        key TEXT PRIMARY KEY, value REAL NOT NULL, updated REAL NOT NULL);
    CREATE TABLE IF NOT EXISTS recommendations (
        key TEXT PRIMARY KEY, severity TEXT NOT NULL, message TEXT NOT NULL,
        first_seen REAL NOT NULL, last_seen REAL NOT NULL, status TEXT NOT NULL DEFAULT 'open');
    CREATE TABLE IF NOT EXISTS audit (
        id INTEGER PRIMARY KEY, ts REAL NOT NULL, event TEXT NOT NULL,
        detail TEXT NOT NULL, prev_hash TEXT NOT NULL, hash TEXT NOT NULL);
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._chmod(path.parent, 0o700)
        self._conn = sqlite3.connect(str(path), check_same_thread=False, timeout=15)
        self._conn.row_factory = sqlite3.Row
        for pragma in (
            "PRAGMA journal_mode=WAL",
            "PRAGMA synchronous=NORMAL",
            "PRAGMA foreign_keys=ON",
            "PRAGMA secure_delete=ON",
            "PRAGMA trusted_schema=OFF",
            "PRAGMA cell_size_check=ON",
        ):
            self._conn.execute(pragma)
        with self._lock, self._conn:
            self._conn.executescript(self.SCHEMA)
        self.lock_down_files()

    @staticmethod
    def _chmod(p: Path, mode: int) -> None:
        try:
            os.chmod(p, mode)
        except OSError:  # e.g. unsupported filesystem; never fatal
            pass

    def lock_down_files(self) -> None:
        for suffix in ("", "-wal", "-shm"):
            p = Path(str(self.path) + suffix)
            if p.exists():
                self._chmod(p, 0o600)

    def integrity_ok(self) -> bool:
        with self._lock:
            row = self._conn.execute("PRAGMA quick_check").fetchone()
        return bool(row) and row[0] == "ok"

    # -- audit ------------------------------------------------------------- #
    def audit(self, event: str, detail: Any = None) -> None:
        detail_s = json.dumps(detail, sort_keys=True, default=str) if detail is not None else ""
        ts = time.time()
        with self._lock, self._conn:
            row = self._conn.execute("SELECT hash FROM audit ORDER BY id DESC LIMIT 1").fetchone()
            prev = row["hash"] if row else GENESIS_HASH
            digest = hashlib.sha256(f"{prev}|{ts!r}|{event}|{detail_s}".encode()).hexdigest()
            self._conn.execute(
                "INSERT INTO audit(ts, event, detail, prev_hash, hash) VALUES (?,?,?,?,?)",
                (ts, event, detail_s, prev, digest),
            )

    def verify_audit(self) -> bool:
        prev = GENESIS_HASH
        with self._lock:
            rows = self._conn.execute("SELECT * FROM audit ORDER BY id").fetchall()
        for r in rows:
            expected = hashlib.sha256(
                f"{prev}|{r['ts']!r}|{r['event']}|{r['detail']}".encode()
            ).hexdigest()
            if r["prev_hash"] != prev or r["hash"] != expected:
                return False
            prev = r["hash"]
        return True

    # -- metrics ----------------------------------------------------------- #
    def record_metrics(self, values: dict[str, float]) -> None:
        ts = time.time()
        with self._lock, self._conn:
            self._conn.executemany(
                "INSERT INTO metrics(ts, name, value) VALUES (?,?,?)",
                [(ts, k, float(v)) for k, v in values.items()],
            )

    def history(self, name: str, limit: int) -> list[float]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT value FROM metrics WHERE name=? ORDER BY ts DESC LIMIT ?", (name, limit)
            ).fetchall()
        return [r["value"] for r in reversed(rows)]

    def metric_names(self) -> list[str]:
        with self._lock:
            return [r[0] for r in self._conn.execute("SELECT DISTINCT name FROM metrics")]

    # -- tunables ---------------------------------------------------------- #
    def get_tunable(self, key: str, default: float) -> float:
        with self._lock:
            row = self._conn.execute("SELECT value FROM tunables WHERE key=?", (key,)).fetchone()
        return float(row["value"]) if row else default

    def set_tunable(self, key: str, value: float) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO tunables(key, value, updated) VALUES (?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated=excluded.updated",
                (key, float(value), time.time()),
            )

    # -- recommendations --------------------------------------------------- #
    def upsert_recommendation(self, key: str, severity: str, message: str) -> None:
        now = time.time()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO recommendations(key, severity, message, first_seen, last_seen) "
                "VALUES (?,?,?,?,?) ON CONFLICT(key) DO UPDATE SET "
                "severity=excluded.severity, message=excluded.message, last_seen=excluded.last_seen",
                (key, severity, message, now, now),
            )

    def resolve_stale_recommendations(self, active_keys: set[str]) -> None:
        with self._lock, self._conn:
            rows = self._conn.execute("SELECT key FROM recommendations WHERE status='open'").fetchall()
            for r in rows:
                if r["key"] not in active_keys:
                    self._conn.execute("UPDATE recommendations SET status='resolved' WHERE key=?", (r["key"],))

    def open_recommendations(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT key, severity, message FROM recommendations WHERE status='open' "
                "ORDER BY CASE severity WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END, key"
            ).fetchall()
        return [dict(r) for r in rows]

    # -- maintenance ------------------------------------------------------- #
    def maintain(self, retention_days: int, vacuum_ratio: float) -> dict[str, Any]:
        cutoff = time.time() - retention_days * 86400
        with self._lock:
            with self._conn:
                pruned = self._conn.execute("DELETE FROM metrics WHERE ts < ?", (cutoff,)).rowcount
            self._conn.execute("PRAGMA optimize")
            self._conn.execute("ANALYZE")
            pages = self._conn.execute("PRAGMA page_count").fetchone()[0] or 1
            free = self._conn.execute("PRAGMA freelist_count").fetchone()[0]
            vacuumed = False
            if free / pages >= vacuum_ratio:
                self._conn.execute("VACUUM")
                vacuumed = True
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.lock_down_files()
        return {"pruned_metrics": pruned, "vacuumed": vacuumed, "free_ratio": round(free / pages, 3)}

    def size_bytes(self) -> int:
        try:
            return self.path.stat().st_size
        except OSError:
            return 0

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            finally:
                self._conn.close()


# --------------------------------------------------------------------------- #
# Project scanner (read-only)
# --------------------------------------------------------------------------- #
@dataclass
class ScanResult:
    metrics: dict[str, float]
    fingerprint: str
    syntax_errors: list[str]
    network_importers: list[str]


def scan_project(cfg: EngineConfig) -> ScanResult:
    files = loc = funcs = classes = todos = test_files = 0
    syntax_errors: list[str] = []
    net_importers: list[str] = []
    digest = hashlib.sha256()

    for dirpath, dirs, names in os.walk(cfg.project_root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED_DIRS)
        for name in sorted(names):
            if not name.endswith(".py"):
                continue
            path = Path(dirpath) / name
            if path.is_symlink() or files >= cfg.max_files:
                continue
            try:
                if path.stat().st_size > cfg.max_file_bytes:
                    continue
                raw = path.read_bytes()
                text = raw.decode("utf-8", errors="replace")
            except OSError:
                continue
            rel = str(path.relative_to(cfg.project_root))
            digest.update(rel.encode())
            digest.update(hashlib.sha256(raw).digest())
            files += 1
            loc += sum(1 for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#"))
            todos += text.count("TODO") + text.count("FIXME")
            if name.startswith("test_") or name.endswith("_test.py") or "tests" in path.parts:
                test_files += 1
            try:
                tree = ast.parse(text, filename=rel)  # parse only; never executed
            except (SyntaxError, ValueError) as exc:
                syntax_errors.append(f"{rel}: {exc.__class__.__name__}")
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    funcs += 1
                elif isinstance(node, ast.ClassDef):
                    classes += 1
                elif isinstance(node, ast.Import):
                    if any(a.name.split(".")[0] in NETWORK_MODULES for a in node.names):
                        net_importers.append(rel)
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.split(".")[0] in NETWORK_MODULES:
                        net_importers.append(rel)

    metrics = {
        "py_files": files,
        "loc": loc,
        "functions": funcs,
        "classes": classes,
        "todo_count": todos,
        "test_files": test_files,
        "syntax_errors": len(syntax_errors),
        "network_import_files": len(set(net_importers)),
    }
    return ScanResult(metrics, digest.hexdigest(), syntax_errors, sorted(set(net_importers)))


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def slope(values: list[float]) -> float:
    n = len(values)
    if n < 3:
        return 0.0
    xs = list(range(n))
    mx, my = statistics.fmean(xs), statistics.fmean(values)
    denom = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, values)) / denom if denom else 0.0


def find_anomalies(store: SecureStore, names: list[str], window: int, z: float) -> list[str]:
    found = []
    for name in names:
        h = store.history(name, window)
        if len(h) < 8:
            continue
        base, latest = h[:-1], h[-1]
        sd = statistics.pstdev(base)
        if sd > 0 and abs(latest - statistics.fmean(base)) / sd >= z:
            found.append(name)
    return found


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class CoreGrowthEngine:
    """Autonomous offline growth engine. Thread-based; safe to start from a FastAPI lifespan."""

    def __init__(self, config: Optional[EngineConfig] = None) -> None:
        self.cfg = config or EngineConfig()
        self.store = SecureStore(self.cfg.db_path)  # type: ignore[arg-type]
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_fingerprint: Optional[str] = None
        self._quiet_cycles = 0
        self._failures = 0
        self._cycles = 0
        self._last_report: dict[str, Any] = {}
        self.interval = self._clamp(self.store.get_tunable("interval", self.cfg.initial_interval))
        self.store.audit("engine_init", {"root": str(self.cfg.project_root), "interval": self.interval})

    # -- lifecycle --------------------------------------------------------- #
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="zentrax-growth-engine", daemon=True)
        self._thread.start()
        log.info("Growth engine started (interval=%.0fs)", self.interval)

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)
        self.store.audit("engine_stop", {"cycles": self._cycles})
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
            except Exception as exc:  # engine must never crash the host app
                self._failures += 1
                log.exception("Growth cycle failed")
                try:
                    self.store.audit("cycle_error", {"error": repr(exc), "failures": self._failures})
                except Exception:
                    pass
            backoff = min(2 ** self._failures, 32) if self._failures else 1
            self._stop.wait(self.interval * backoff)

    # -- one full cycle ---------------------------------------------------- #
    def run_cycle(self) -> dict[str, Any]:
        if self.kill_switch_active():
            self.store.audit("kill_switch_active")
            self._last_report = {"status": "halted_by_kill_switch"}
            return self._last_report

        if not self.store.integrity_ok():
            self.store.audit("db_integrity_failure")
            self._last_report = {"status": "halted_db_integrity_failure"}
            log.error("Database integrity check failed; engine paused.")
            return self._last_report

        audit_ok = self.store.verify_audit()
        if not audit_ok:
            self.store.audit("audit_chain_broken")

        scan = scan_project(self.cfg)
        self.store.record_metrics({**scan.metrics, "db_size_bytes": self.store.size_bytes()})

        anomalies = find_anomalies(
            self.store, list(scan.metrics), self.cfg.history_window, self.cfg.anomaly_zscore
        )
        trends = {n: round(slope(self.store.history(n, self.cfg.history_window)), 4)
                  for n in ("loc", "todo_count", "functions", "db_size_bytes")}

        tuning = self._self_optimize(scan)
        maintenance = self.store.maintain(self.cfg.retention_days, self.cfg.fragmentation_vacuum_ratio)
        recs = self._recommend(scan, trends, anomalies, audit_ok)

        self._cycles += 1
        self._last_report = {
            "status": "ok",
            "cycle": self._cycles,
            "metrics": scan.metrics,
            "trends": trends,
            "anomalies": anomalies,
            "tuning": tuning,
            "maintenance": maintenance,
            "open_recommendations": len(recs),
            "audit_chain_valid": audit_ok,
        }
        self.store.audit("cycle_complete", {k: self._last_report[k] for k in ("cycle", "anomalies", "tuning", "maintenance")})
        return self._last_report

    # -- self-optimization (bounded parameters only) ----------------------- #
    def _self_optimize(self, scan: ScanResult) -> dict[str, Any]:
        changed = scan.fingerprint != self._last_fingerprint
        self._last_fingerprint = scan.fingerprint
        old = self.interval
        if changed:
            self._quiet_cycles = 0
            self.interval = self._clamp(self.interval * 0.5)      # project is active: look more often
        else:
            self._quiet_cycles += 1
            if self._quiet_cycles >= 3:
                self.interval = self._clamp(self.interval * 1.5)  # project is idle: save resources
        if self.interval != old:
            self.store.set_tunable("interval", self.interval)
        return {"project_changed": changed, "interval": round(self.interval, 1)}

    # -- growth recommendations (never edits source) ----------------------- #
    def _recommend(self, scan: ScanResult, trends: dict[str, float],
                   anomalies: list[str], audit_ok: bool) -> list[dict[str, Any]]:
        m = scan.metrics
        active: set[str] = set()

        def add(key: str, sev: str, msg: str) -> None:
            active.add(key)
            self.store.upsert_recommendation(key, sev, msg)

        if not audit_ok:
            add("audit_chain", "high", "Audit log hash chain is broken: the local database may have been tampered with.")
        if scan.syntax_errors:
            add("syntax_errors", "high", f"{len(scan.syntax_errors)} file(s) fail to parse: {', '.join(scan.syntax_errors[:5])}")
        if m["py_files"] and m["test_files"] / m["py_files"] < 0.15:
            add("test_coverage", "medium", "Test files are under 15% of Python files; add tests for core modules.")
        if m["todo_count"] > 25 or trends.get("todo_count", 0) > 0.5:
            add("todo_growth", "medium", f"{int(m['todo_count'])} TODO/FIXME markers and rising; schedule a cleanup pass.")
        if m["functions"] and m["loc"] / max(m["functions"], 1) > 40:
            add("large_functions", "low", "Average function size is high; consider splitting large functions.")
        if scan.network_importers:
            add("network_imports", "low",
                "Modules importing network libraries (verify each is gated by your offline kill switch / fallback router): "
                + ", ".join(scan.network_importers[:8]))
        for name in anomalies:
            add(f"anomaly_{name}", "medium", f"Metric '{name}' changed abnormally versus its recent history.")

        self.store.resolve_stale_recommendations(active)
        return self.store.open_recommendations()

    # -- public read API (for FastAPI routes) ------------------------------ #
    def status(self) -> dict[str, Any]:
        return {
            "running": bool(self._thread and self._thread.is_alive()),
            "kill_switch": self.kill_switch_active(),
            "interval_seconds": round(self.interval, 1),
            "cycles": self._cycles,
            "last_report": self._last_report,
        }

    def recommendations(self) -> list[dict[str, Any]]:
        return self.store.open_recommendations()


# --------------------------------------------------------------------------- #
# Singleton helper + CLI
# --------------------------------------------------------------------------- #
_engine: Optional[CoreGrowthEngine] = None
_engine_lock = threading.Lock()


def get_engine(config: Optional[EngineConfig] = None) -> CoreGrowthEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = CoreGrowthEngine(config)
        return _engine


def main() -> None:
    parser = argparse.ArgumentParser(description="ZentraX-AI core growth engine (offline)")
    parser.add_argument("--once", action="store_true", help="run a single cycle and print the report")
    parser.add_argument("--root", type=Path, default=None, help="project root (defaults to repo root)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = EngineConfig(project_root=args.root) if args.root else EngineConfig()
    engine = CoreGrowthEngine(cfg)
    try:
        if args.once:
            print(json.dumps(engine.run_cycle(), indent=2, default=str))
            print(json.dumps(engine.recommendations(), indent=2))
        else:
            engine.start()
            while True:
                time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
