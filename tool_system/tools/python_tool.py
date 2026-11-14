"""Python code execution tool (sandboxed subprocess).

Each run happens in a **separate interpreter process** (``python -I``) with:
CPU/memory/file-size/fd rlimits (POSIX), a scrubbed environment, a throw-away working
directory, its own process group (killed on timeout), output captured via size-limited
files, and a runtime import blocker for dangerous modules (network, subprocess, ctypes ...).

This protects the host from runaway or careless code.  It is NOT a hardened boundary against a
determined attacker - for untrusted code use containers/VMs/gVisor.

Config keys: ``timeout`` (s), ``cpu_seconds``, ``memory_mb``, ``max_output_bytes``,
``blocked_modules`` (list), ``python`` (interpreter path).
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List

from ..tool_core import (
    BaseTool, ToolCancelledError, ToolCategory, ToolContext, ToolExecutionError, ToolParameter,
    ToolSpec, ToolTimeoutError,
)

try:
    import resource
except ImportError:  # pragma: no cover - Windows
    resource = None  # type: ignore[assignment]

DEFAULT_BLOCKED = ("subprocess", "socket", "_socket", "ssl", "ctypes", "_ctypes", "multiprocessing",
                   "_posixsubprocess", "pty", "http", "urllib", "ftplib", "smtplib", "telnetlib")

_RUNNER = r'''
import sys, json, runpy, importlib.abc
BLOCKED = frozenset(json.loads(sys.argv[1]))
class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError("import of '%s' is blocked in this sandbox" % name)
        return None
for _m in list(sys.modules):
    if _m.split(".")[0] in BLOCKED:
        del sys.modules[_m]
sys.meta_path.insert(0, _Blocker())
sys.argv = ["code.py"]
runpy.run_path("code.py", run_name="__main__")
'''


class PythonTool(BaseTool):
    """Runs a Python snippet in a resource-limited subprocess and returns its output."""

    spec = ToolSpec(
        name="python_exec",
        description=("Execute Python 3 code and return stdout, stderr and the exit code. Print what "
                     "you want to see. No network or subprocess access; limited CPU, memory and time."),
        parameters=(
            ToolParameter("code", "string", "Python source code to run.", max_length=100_000),
            ToolParameter("timeout", "number", "Wall-clock limit in seconds.", required=False,
                          default=10, minimum=1, maximum=120),
        ),
        category=ToolCategory.CODE, tags=("code", "python"),
        required_permissions=frozenset({"code.exec"}), timeout=130.0,
    )

    def run(self, context: ToolContext, code: str, timeout: float = 10, **_: Any) -> Any:
        cfg = self.config
        limit = min(float(timeout), float(cfg.get("timeout", 120)))
        remaining = context.remaining()
        if remaining is not None:
            limit = min(limit, max(0.1, remaining - 1))
        cap = int(cfg.get("max_output_bytes", 1_000_000))
        blocked = list(cfg.get("blocked_modules", DEFAULT_BLOCKED))
        python = cfg.get("python", sys.executable)
        workdir = tempfile.mkdtemp(prefix="pytool_")
        started = time.monotonic()
        try:
            with open(os.path.join(workdir, "code.py"), "w", encoding="utf-8") as fh:
                fh.write(code)
            with open(os.path.join(workdir, "_runner.py"), "w", encoding="utf-8") as fh:
                fh.write(_RUNNER)
            out_path, err_path = os.path.join(workdir, ".stdout"), os.path.join(workdir, ".stderr")
            with open(out_path, "wb") as out, open(err_path, "wb") as err:
                try:
                    proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
                        [python, "-I", "_runner.py", json.dumps(blocked)], cwd=workdir,
                        stdin=subprocess.DEVNULL, stdout=out, stderr=err, env=self._env(workdir),
                        start_new_session=(os.name == "posix"), preexec_fn=self._limiter(cap))
                except (OSError, subprocess.SubprocessError) as exc:
                    raise ToolExecutionError(f"could not start interpreter: {exc}") from exc
                timed_out = False
                try:
                    while proc.poll() is None:
                        context.check_cancelled()
                        if time.monotonic() - started > limit:
                            timed_out = True
                            break
                        time.sleep(0.02)
                except (ToolCancelledError, ToolTimeoutError):
                    self._kill(proc)
                    raise
                if timed_out:
                    self._kill(proc)
            stdout, t1 = self._read(out_path, cap)
            stderr, t2 = self._read(err_path, cap)
            code_ = proc.returncode
            result: Dict[str, Any] = {
                "stdout": stdout, "stderr": stderr, "exit_code": code_, "timed_out": timed_out,
                "duration": round(time.monotonic() - started, 3), "truncated": t1 or t2}
            if code_ is not None and code_ < 0 and not timed_out:
                result["terminated_by_signal"] = signal.Signals(-code_).name if -code_ in signal.Signals._value2member_map_ else -code_
            return result
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    # -- helpers ------------------------------------------------------------ #
    @staticmethod
    def _env(workdir: str) -> Dict[str, str]:
        env = {k: os.environ[k] for k in ("PATH", "LANG", "LC_ALL", "TZ") if k in os.environ}
        env.update(HOME=workdir, TMPDIR=workdir, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
        return env

    def _limiter(self, max_file: int):
        """Build a ``preexec_fn`` applying rlimits (best effort per limit; none on Windows)."""
        if resource is None:
            return None
        cpu = int(self.config.get("cpu_seconds", 10))
        mem = int(float(self.config.get("memory_mb", 512)) * 1024 * 1024)
        fsize = max(max_file * 2, 1_000_000)

        def apply() -> None:
            def lim(name: str, soft: int, hard: int) -> None:
                rid = getattr(resource, name, None)
                if rid is None:
                    return
                try:
                    _, cur_hard = resource.getrlimit(rid)
                    if cur_hard != resource.RLIM_INFINITY:
                        hard, soft = min(hard, cur_hard), min(soft, cur_hard)
                    resource.setrlimit(rid, (soft, hard))
                except (ValueError, OSError):
                    pass  # e.g. RLIMIT_AS unsupported on macOS
            lim("RLIMIT_CPU", cpu, cpu + 1)
            lim("RLIMIT_AS", mem, mem)
            lim("RLIMIT_FSIZE", fsize, fsize)
            lim("RLIMIT_NOFILE", 64, 64)
            lim("RLIMIT_CORE", 0, 0)
        return apply

    @staticmethod
    def _kill(proc: "subprocess.Popen[bytes]") -> None:
        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGKILL)
            else:  # pragma: no cover
                proc.kill()
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover
            pass

    @staticmethod
    def _read(path: str, cap: int) -> "tuple[str, bool]":
        with open(path, "rb") as fh:
            data = fh.read(cap + 1)
        return data[:cap].decode("utf-8", errors="replace"), len(data) > cap
