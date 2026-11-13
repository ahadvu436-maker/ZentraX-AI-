"""Process-level sandbox: run callables or commands under resource limits.

Isolation model: every run happens in a *separate OS process* with rlimits
applied (see :mod:`.resource_limits`), a scrubbed environment, a throw-away
working directory, its own process group (so the whole tree can be killed) and a
hard wall-clock deadline.  This protects the host process from runaway CPU,
memory, disk and hangs; it is **not** a security boundary against malicious code
(use containers/VMs/seccomp for that).

Callables: with the default ``fork`` start method (POSIX) any callable works;
with ``spawn``/``forkserver`` it must be picklable.  Return values must be
picklable.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import multiprocessing
import os
import shutil
import signal
import subprocess
import tempfile
import time
import traceback
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Sequence

from ..execution_core import (
    ResourceLimitError, SandboxError, TaskFailedError, TaskTimeoutError, run_awaitable_blocking,
)
from .resource_limits import ResourceLimits

logger = logging.getLogger(__name__)

__all__ = ["ExecutionSandbox", "SandboxResult", "terminate_process_tree"]

_POSIX = os.name == "posix"
_SIGXCPU = getattr(signal, "SIGXCPU", None)
_SIGXFSZ = getattr(signal, "SIGXFSZ", None)
_SIGKILL = getattr(signal, "SIGKILL", signal.SIGTERM)
_SIGTERM = signal.SIGTERM


def _limit_reason(returncode: Optional[int]) -> Optional[str]:
    """Translate a negative return code into a resource-limit reason, if any."""
    if returncode is None or returncode >= 0:
        return None
    sig = -returncode
    if _SIGXCPU is not None and sig == _SIGXCPU:
        return "cpu_time"
    if _SIGXFSZ is not None and sig == _SIGXFSZ:
        return "file_size"
    if sig == _SIGKILL:
        return "killed (CPU hard limit or memory pressure)"
    return None


def terminate_process_tree(proc: "subprocess.Popen[Any]", grace: float = 0.0) -> None:
    """SIGTERM (optional grace period) then SIGKILL the process group of ``proc``."""
    def _signal(sig: int) -> None:
        try:
            if _POSIX:
                os.killpg(proc.pid, sig)  # group id == pid thanks to start_new_session
            elif sig == _SIGKILL:
                proc.kill()
            else:
                proc.terminate()
        except (ProcessLookupError, PermissionError):
            pass
        except OSError:
            try:
                proc.kill()
            except OSError:
                pass

    if proc.poll() is None and grace > 0:
        _signal(_SIGTERM)
        deadline = time.monotonic() + grace
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
    if proc.poll() is None or _POSIX:
        _signal(_SIGKILL)  # also reaps stragglers left in the group
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:  # pragma: no cover
        logger.error("process %s did not die after SIGKILL", proc.pid)


@dataclass
class SandboxResult:
    """Outcome of :meth:`ExecutionSandbox.run_command`."""

    returncode: Optional[int]
    stdout: str
    stderr: str
    duration: float
    timed_out: bool = False
    limit_exceeded: Optional[str] = None
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def _child_main(conn: Any, func: Callable[..., Any], args: Sequence[Any],
                kwargs: Dict[str, Any], limits: Optional[ResourceLimits],
                workdir: Optional[str]) -> None:
    """Entry point of the sandboxed child process (module-level so it pickles)."""
    try:
        if _POSIX:
            try:
                os.setpgrp()
            except OSError:
                pass
        if workdir:
            os.chdir(workdir)
        if limits is not None:
            limits.apply()
        value = func(*args, **kwargs)
        if inspect.isawaitable(value):
            value = run_awaitable_blocking(value)
        payload: tuple = ("ok", value)
    except MemoryError:
        payload = ("limit", "memory", "MemoryError inside sandbox")
    except SandboxError as exc:
        payload = ("limit", "setup", str(exc))
    except BaseException as exc:  # noqa: BLE001 - everything must be reported
        payload = ("error", type(exc).__name__, str(exc), traceback.format_exc())
    try:
        conn.send(payload)
    except Exception as exc:  # noqa: BLE001 - e.g. unpicklable result
        try:
            conn.send(("error", type(exc).__name__, f"result could not be sent: {exc}", ""))
        except Exception:  # noqa: BLE001
            pass
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


class ExecutionSandbox:
    """Runs work in resource-limited child processes."""

    def __init__(
        self,
        default_limits: Optional[ResourceLimits] = None,
        *,
        base_dir: Optional[str] = None,
        use_temp_cwd: bool = True,
        env_allowlist: Sequence[str] = ("PATH", "LANG", "LC_ALL", "TZ"),
        start_method: Optional[str] = None,
        max_output_bytes: int = 1024 * 1024,
    ) -> None:
        self.default_limits = default_limits or ResourceLimits()
        self.base_dir = base_dir
        self.use_temp_cwd = use_temp_cwd
        self.env_allowlist = tuple(env_allowlist)
        self.max_output_bytes = max_output_bytes
        if start_method is None:
            start_method = "fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn"
        self.start_method = start_method

    # -- helpers ------------------------------------------------------------ #
    @contextmanager
    def workdir(self) -> Iterator[Optional[str]]:
        """Temporary working directory, removed on exit (``None`` if disabled)."""
        if not self.use_temp_cwd:
            yield None
            return
        path = tempfile.mkdtemp(prefix="sandbox_", dir=self.base_dir)
        try:
            yield path
        finally:
            shutil.rmtree(path, ignore_errors=True)

    def build_env(self, extra: Optional[Mapping[str, str]] = None,
                  home: Optional[str] = None) -> Dict[str, str]:
        """Scrubbed environment: allow-listed variables only, HOME/TMPDIR pinned."""
        env = {k: os.environ[k] for k in self.env_allowlist if k in os.environ}
        if home:
            env["HOME"] = env["TMPDIR"] = home
        if extra:
            env.update({str(k): str(v) for k, v in extra.items()})
        return env

    @staticmethod
    def _effective_timeout(timeout: Optional[float], limits: ResourceLimits) -> Optional[float]:
        candidates = [t for t in (timeout, limits.wall_time_seconds) if t is not None]
        return min(candidates) if candidates else None

    def _check_platform(self, limits: ResourceLimits) -> None:
        if limits.has_rlimits() and limits.require_enforcement and not ResourceLimits.supported():
            raise SandboxError("resource limits cannot be enforced on this platform")

    # -- callables ---------------------------------------------------------- #
    def run_callable(
        self, func: Callable[..., Any], args: Sequence[Any] = (),
        kwargs: Optional[Dict[str, Any]] = None, *,
        limits: Optional[ResourceLimits] = None, timeout: Optional[float] = None,
    ) -> Any:
        """Run ``func(*args, **kwargs)`` in a limited child process and return its value.

        Raises:
            TaskTimeoutError: wall-clock deadline exceeded (process tree killed).
            ResourceLimitError: CPU/memory/file-size limit hit or unenforceable.
            TaskFailedError: ``func`` raised; message includes the remote traceback.
            SandboxError: the child crashed or the setup failed.
        """
        if not callable(func):
            raise TypeError("func must be callable")
        limits = limits or self.default_limits
        self._check_platform(limits)
        effective = self._effective_timeout(timeout, limits)
        ctx = multiprocessing.get_context(self.start_method)
        recv_conn, send_conn = ctx.Pipe(duplex=False)

        with self.workdir() as cwd:
            proc = ctx.Process(
                target=_child_main,
                args=(send_conn, func, tuple(args), dict(kwargs or {}), limits, cwd),
                daemon=True,
            )
            try:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", DeprecationWarning)  # fork + threads
                        proc.start()
                except Exception as exc:  # noqa: BLE001
                    raise SandboxError(
                        f"could not start sandbox process ({self.start_method}): {exc}") from exc
                finally:
                    send_conn.close()
                payload = self._wait_payload(proc, recv_conn, effective)
                return self._interpret(proc, payload)
            finally:
                self._reap(proc)
                recv_conn.close()

    def _wait_payload(self, proc: Any, conn: Any, timeout: Optional[float]) -> Optional[tuple]:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                self._kill(proc)
                raise TaskTimeoutError(f"sandboxed task exceeded {timeout:.2f}s wall-clock limit")
            slice_ = 0.05 if remaining is None else min(0.05, remaining)
            try:
                if conn.poll(slice_):
                    return conn.recv()
                if not proc.is_alive():
                    return conn.recv() if conn.poll(0.2) else None
            except (EOFError, OSError):
                return None  # child died without reporting
            except Exception as exc:  # noqa: BLE001 - corrupt payload
                raise SandboxError(f"could not read sandbox result: {exc}") from exc

    def _interpret(self, proc: Any, payload: Optional[tuple]) -> Any:
        proc.join(1)
        if payload is None:
            code = proc.exitcode
            reason = _limit_reason(code)
            if reason:
                raise ResourceLimitError(f"sandbox process terminated: {reason} (exit {code})")
            raise SandboxError(f"sandbox process exited (code {code}) without returning a result")
        kind = payload[0]
        if kind == "ok":
            return payload[1]
        if kind == "limit":
            raise ResourceLimitError(f"{payload[1]} limit: {payload[2]}")
        _, exc_type, message, tb = payload
        err = TaskFailedError(f"{exc_type}: {message}\n--- sandbox traceback ---\n{tb}".rstrip())
        err.remote_type = exc_type
        raise err

    def _kill(self, proc: Any) -> None:
        pid = proc.pid
        if _POSIX and pid:
            try:
                os.killpg(pid, _SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        if proc.is_alive():
            proc.kill()
        proc.join(2)

    def _reap(self, proc: Any) -> None:
        try:
            if proc.is_alive():
                self._kill(proc)
            else:
                proc.join(0.5)
            proc.close()
        except (ValueError, OSError):
            pass

    # -- commands ----------------------------------------------------------- #
    def popen(
        self, argv: Sequence[str], *, cwd: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None, limits: Optional[ResourceLimits] = None,
        stdin: Any = subprocess.DEVNULL, stdout: Any = subprocess.PIPE,
        stderr: Any = subprocess.PIPE,
    ) -> "subprocess.Popen[bytes]":
        """Start ``argv`` (never via a shell) with limits, a scrubbed env and its own session.

        Note: on POSIX the limits are applied via ``preexec_fn`` which Python documents as
        unsafe when other threads hold locks at fork time; keep the command short-lived to
        reach ``exec`` quickly (it is, since only ``setrlimit`` calls run in between).
        """
        if not argv or not all(isinstance(a, str) for a in argv):
            raise ValueError("argv must be a non-empty sequence of strings")
        limits = limits or self.default_limits
        self._check_platform(limits)
        kwargs: Dict[str, Any] = {}
        if _POSIX:
            kwargs["start_new_session"] = True
            if limits.has_rlimits():
                kwargs["preexec_fn"] = limits.apply
        try:
            return subprocess.Popen(  # noqa: S603 - shell=False by construction
                list(argv), cwd=cwd, env=self.build_env(env, home=cwd), stdin=stdin,
                stdout=stdout, stderr=stderr, shell=False, **kwargs)
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            raise SandboxError(f"could not start {argv[0]!r}: {exc}") from exc

    def run_command(
        self, argv: Sequence[str], *, input_text: Optional[str] = None,
        limits: Optional[ResourceLimits] = None, timeout: Optional[float] = None,
        env: Optional[Mapping[str, str]] = None, cwd: Optional[str] = None, check: bool = False,
    ) -> SandboxResult:
        """Run a command to completion under limits and return its output.

        With ``check=True`` raises :class:`TaskTimeoutError` on timeout and
        :class:`ResourceLimitError` / :class:`TaskFailedError` on a bad exit.
        """
        limits = limits or self.default_limits
        effective = self._effective_timeout(timeout, limits)
        start = time.monotonic()
        with self.workdir() as tmp:
            run_cwd = cwd or tmp
            proc = self.popen(
                argv, cwd=run_cwd, env=env, limits=limits,
                stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL)
            timed_out = False
            try:
                out, err = proc.communicate(
                    input=input_text.encode() if input_text is not None else None, timeout=effective)
            except subprocess.TimeoutExpired:
                timed_out = True
                terminate_process_tree(proc)
                out, err = proc.communicate()
            except BaseException:
                terminate_process_tree(proc)
                raise
            finally:
                if _POSIX:
                    terminate_process_tree(proc)  # reap any stragglers in the group

        cap = self.max_output_bytes
        truncated = len(out) > cap or len(err) > cap
        result = SandboxResult(
            returncode=proc.returncode,
            stdout=out[:cap].decode(errors="replace"), stderr=err[:cap].decode(errors="replace"),
            duration=time.monotonic() - start, timed_out=timed_out,
            limit_exceeded=_limit_reason(proc.returncode), truncated=truncated)
        if check:
            if timed_out:
                raise TaskTimeoutError(f"command {argv[0]!r} exceeded {effective:.2f}s")
            if result.limit_exceeded:
                raise ResourceLimitError(f"command {argv[0]!r}: {result.limit_exceeded}")
            if result.returncode != 0:
                raise TaskFailedError(
                    f"command {argv[0]!r} exited with {result.returncode}: {result.stderr[-500:]}")
        return result
