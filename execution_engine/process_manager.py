"""Tracking and lifecycle management of sandboxed OS processes."""
from __future__ import annotations

import logging
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import IO, Any, Dict, List, Optional, Sequence

from .execution_core import ProcessError, TaskTimeoutError
from .execution_monitor import ExecutionMonitor
from .sandbox.execution_sandbox import ExecutionSandbox, SandboxResult, _limit_reason, terminate_process_tree
from .sandbox.resource_limits import ResourceLimits

logger = logging.getLogger(__name__)

__all__ = ["ManagedProcess", "ProcessManager"]


@dataclass
class ManagedProcess:
    """Handle for a process started through :class:`ProcessManager`."""

    name: str
    argv: List[str]
    popen: Any
    started_at: float
    deadline: Optional[float]
    workdir_ctx: Any
    stdout: IO[bytes]
    stderr: IO[bytes]
    timed_out: bool = False
    killed: bool = False
    _result: Optional[SandboxResult] = field(default=None, repr=False)

    @property
    def pid(self) -> int:
        return self.popen.pid

    @property
    def running(self) -> bool:
        return self.popen.poll() is None

    @property
    def returncode(self) -> Optional[int]:
        return self.popen.poll()


class ProcessManager:
    """Starts, tracks, waits for and terminates sandboxed processes.

    Output goes to temporary files (not pipes) so a chatty process can never
    block on a full pipe.  A watchdog thread enforces per-process deadlines.
    """

    def __init__(self, sandbox: ExecutionSandbox, monitor: Optional[ExecutionMonitor] = None,
                 max_processes: int = 16) -> None:
        self.sandbox = sandbox
        self.monitor = monitor
        self.max_processes = max_processes
        self._procs: Dict[int, ManagedProcess] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle ---------------------------------------------------------- #
    def spawn(
        self, argv: Sequence[str], *, name: Optional[str] = None,
        limits: Optional[ResourceLimits] = None, timeout: Optional[float] = None,
        env: Optional[Dict[str, str]] = None, cwd: Optional[str] = None,
    ) -> ManagedProcess:
        """Start ``argv`` under sandbox limits; raises :class:`ProcessError` when at capacity."""
        self.reap()
        with self._lock:
            if len(self._procs) >= self.max_processes:
                raise ProcessError(f"process limit reached ({self.max_processes})")
        limits = limits or self.sandbox.default_limits
        wall = [t for t in (timeout, limits.wall_time_seconds) if t is not None]
        deadline = time.monotonic() + min(wall) if wall else None

        workdir_ctx = self.sandbox.workdir()
        tmp = workdir_ctx.__enter__()
        out, err = tempfile.TemporaryFile(), tempfile.TemporaryFile()
        try:
            popen = self.sandbox.popen(argv, cwd=cwd or tmp, env=env, limits=limits,
                                       stdout=out, stderr=err)
        except BaseException:
            out.close()
            err.close()
            workdir_ctx.__exit__(None, None, None)
            raise
        proc = ManagedProcess(name or str(argv[0]), list(argv), popen, time.monotonic(),
                              deadline, workdir_ctx, out, err)
        with self._lock:
            self._procs[popen.pid] = proc
        if self.monitor:
            self.monitor.emit("process_started", str(popen.pid), proc.name, argv=list(argv))
        return proc

    def get(self, pid: int) -> ManagedProcess:
        with self._lock:
            try:
                return self._procs[pid]
            except KeyError:
                raise ProcessError(f"unknown process {pid}") from None

    def list(self) -> List[ManagedProcess]:
        with self._lock:
            return list(self._procs.values())

    def wait(self, pid: int, timeout: Optional[float] = None) -> SandboxResult:
        """Wait for a process, collect its output and release its resources."""
        proc = self.get(pid)
        if proc._result is not None:
            return proc._result
        limit = timeout
        if proc.deadline is not None:
            remaining = max(0.0, proc.deadline - time.monotonic())
            limit = remaining if limit is None else min(limit, remaining)
        try:
            proc.popen.wait(timeout=limit)
        except Exception:  # subprocess.TimeoutExpired
            if proc.deadline is not None and time.monotonic() >= proc.deadline:
                proc.timed_out = True
                terminate_process_tree(proc.popen)
            else:
                raise TaskTimeoutError(f"wait for process {pid} timed out after {timeout}s") from None
        return self._finalize(proc)

    def terminate(self, pid: int, grace: float = 3.0) -> None:
        """Politely stop (SIGTERM), then kill, the process group and release it."""
        proc = self.get(pid)
        proc.killed = True
        terminate_process_tree(proc.popen, grace=grace)
        self._finalize(proc)

    def terminate_all(self, grace: float = 1.0) -> None:
        for proc in self.list():
            try:
                self.terminate(proc.pid, grace=grace)
            except Exception:  # noqa: BLE001
                logger.exception("failed to terminate process %s", proc.pid)

    def reap(self) -> List[SandboxResult]:
        """Enforce deadlines and release finished processes; returns their results."""
        results = []
        for proc in self.list():
            if proc._result is not None:
                continue
            if proc.running and proc.deadline is not None and time.monotonic() >= proc.deadline:
                proc.timed_out = True
                logger.warning("process %s exceeded its deadline; killing", proc.pid)
                terminate_process_tree(proc.popen)
            if not proc.running:
                results.append(self._finalize(proc))
        return results

    # -- watchdog ----------------------------------------------------------- #
    def start_watchdog(self, interval: float = 1.0) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._watch, args=(interval,),
                                            daemon=True, name="process-watchdog")
            self._thread.start()

    def _watch(self, interval: float) -> None:
        while not self._stop.wait(interval):
            try:
                self.reap_deadlines_only()
            except Exception:  # noqa: BLE001
                logger.exception("process watchdog error")

    def reap_deadlines_only(self) -> None:
        """Kill overdue processes but keep finished ones until ``wait()`` collects them."""
        for proc in self.list():
            if proc.running and proc.deadline is not None and time.monotonic() >= proc.deadline:
                proc.timed_out = True
                logger.warning("process %s exceeded its deadline; killing", proc.pid)
                terminate_process_tree(proc.popen)

    def shutdown(self) -> None:
        self._stop.set()
        self.terminate_all(grace=0.5)
        thread = self._thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2)

    # -- internals ----------------------------------------------------------- #
    def _finalize(self, proc: ManagedProcess) -> SandboxResult:
        with self._lock:
            if proc._result is not None:
                return proc._result
            self._procs.pop(proc.pid, None)
        cap = self.sandbox.max_output_bytes
        streams = []
        for fh in (proc.stdout, proc.stderr):
            try:
                size = fh.seek(0, 2)
                fh.seek(0)
                streams.append((fh.read(cap).decode(errors="replace"), size > cap))
            except (OSError, ValueError):
                streams.append(("", False))
            finally:
                try:
                    fh.close()
                except OSError:
                    pass
        try:
            proc.workdir_ctx.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            logger.exception("workdir cleanup failed")
        code = proc.popen.returncode
        proc._result = SandboxResult(
            returncode=code, stdout=streams[0][0], stderr=streams[1][0],
            duration=time.monotonic() - proc.started_at, timed_out=proc.timed_out,
            limit_exceeded=None if proc.killed or proc.timed_out else _limit_reason(code),
            truncated=streams[0][1] or streams[1][1])
        if self.monitor:
            self.monitor.emit("process_finished", str(proc.pid), proc.name, returncode=code,
                              timed_out=proc.timed_out)
        return proc._result
