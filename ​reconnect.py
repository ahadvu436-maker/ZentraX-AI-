#!/usr/bin/env python3
"""
reconnect.py - verify the host is safe and healthy, then lift the kill switch
(firewall rules / downed interfaces) and reconnect to the global network
automatically. Companion to kill_switch.sh. Standard library only.

Flow (per cycle, only when the host is currently isolated)
-----------------------------------------------------------
 0. Gates      HOLD file present -> stay isolated; too many reconnects in the
               last hour -> stay isolated (anti-flapping)
 1. Checks     resources, required local services, listening-port allowlist,
               no extra UID-0 accounts, file-integrity baseline.
               ALL must pass `stable_passes` times in a row.
 2. Restore    `kill_switch.sh off` (or direct nft/iptables cleanup), bring
               interfaces up, wait for link + default route.
 3. Verify     DNS + TCP reachability of the probe targets.
 4. Re-check   run the safety checks again after `post_check_delay`; if they
               now fail, RE-ISOLATE automatically (kill_switch.sh on --force).

Modes
-----
  reconnect.py --init-baseline        record file hashes (run while KNOWN GOOD)
  reconnect.py [--once]               one cycle (default)
  reconnect.py --watch                run forever, reconnect whenever safe
  reconnect.py --dry-run              run checks only; change nothing
  --config FILE.json                  override any Config field (see below)

Exit codes: 0 reconnected / nothing to do / dry-run ok, 1 stayed isolated,
            3 reconnected but re-isolated after failed re-check.

Important: a host's own checks cannot prove the host is uncompromised. Create
the HOLD file (default /var/lib/reconnect/HOLD) when isolation was triggered
by a person or a security incident; the script will never override it.

systemd example (/etc/systemd/system/reconnect.service):
  [Service]
  ExecStart=/usr/bin/python3 /opt/ops/reconnect.py --watch --config /etc/reconnect.json
  Restart=always
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import logging.handlers
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field, fields
from pathlib import Path

log = logging.getLogger("reconnect")


# ======================================================================
# Configuration
# ======================================================================
@dataclass
class Config:
    kill_switch: str = str(Path(__file__).resolve().with_name("kill_switch.sh"))
    interfaces: list = field(default_factory=list)       # e.g. ["eth0"]; brought up on restore
    state_dir: str = "/var/lib/reconnect"
    hold_file: str = ""                                   # default: <state_dir>/HOLD
    stable_passes: int = 3                                # consecutive clean passes required
    check_interval: int = 20                              # seconds between passes / watch polls
    max_reconnects_per_hour: int = 3
    post_check_delay: int = 15
    verify_timeout: int = 60
    # --- health thresholds
    min_disk_free_pct: float = 10.0
    min_mem_available_pct: float = 5.0
    max_load_per_cpu: float = 4.0
    # --- security checks
    required_services: list = field(default_factory=list)  # ["127.0.0.1:5432"]
    allowed_listen_ports: list = None                      # None = skip; [] = nothing may listen
    check_uid0: bool = True
    integrity_paths: list = field(default_factory=lambda: [
        "/etc/passwd", "/etc/ssh/sshd_config", "/root/.ssh/authorized_keys"])
    # --- connectivity verification
    probe_targets: list = field(default_factory=lambda: ["1.1.1.1:443", "8.8.8.8:53"])
    dns_probe_host: str = "example.com"

    def __post_init__(self):
        if not self.hold_file:
            self.hold_file = str(Path(self.state_dir) / "HOLD")


def load_config(path):
    cfg = {}
    if path:
        with open(path) as f:
            cfg = json.load(f)
    known = {f.name for f in fields(Config)}
    unknown = set(cfg) - known
    if unknown:
        raise SystemExit(f"unknown config keys: {sorted(unknown)}")
    return Config(**cfg)


# ======================================================================
# Small helpers (none of these raise)
# ======================================================================
def run(cmd, timeout=30):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).strip()
    except (OSError, subprocess.SubprocessError) as e:
        return 127, str(e)


def ks_cmd(cfg, *args):
    base = ["bash", cfg.kill_switch] if cfg.kill_switch.endswith(".sh") else [cfg.kill_switch]
    return base + list(args)


def tcp_ok(target, timeout=3.0):
    try:
        host, port = target.rsplit(":", 1)
        with socket.create_connection((host.strip("[]"), int(port)), timeout=timeout):
            return True
    except (OSError, ValueError):
        return False


class State:
    """Persists reconnect timestamps (anti-flapping). Falls back to memory."""

    def __init__(self, directory):
        self.path = Path(directory) / "state.json"
        self.reconnects = []
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            self.reconnects = json.loads(self.path.read_text()).get("reconnects", [])
        except (OSError, ValueError):
            pass

    def recent(self, window=3600):
        now = time.time()
        return [t for t in self.reconnects if now - t < window]

    def record(self):
        self.reconnects = self.recent() + [time.time()]
        try:
            self.path.write_text(json.dumps({"reconnects": self.reconnects}))
        except OSError as e:
            log.warning("could not persist state: %s", e)


# ======================================================================
# Isolation state
# ======================================================================
def rules_active(cfg):
    if Path(cfg.kill_switch).exists():
        _, out = run(ks_cmd(cfg, "status"))
        return "ACTIVE" in out
    if shutil.which("nft") and run(["nft", "list", "table", "inet", "killswitch"])[0] == 0:
        return True
    if shutil.which("iptables") and run(["iptables", "-w", "-C", "INPUT", "-j", "KS_IN"])[0] == 0:
        return True
    return False


def ifaces_down(cfg):
    for ifc in cfg.interfaces:
        try:
            if Path(f"/sys/class/net/{ifc}/operstate").read_text().strip() == "down":
                return True
        except OSError:
            pass
    return False


def is_isolated(cfg):
    return rules_active(cfg) or ifaces_down(cfg)


# ======================================================================
# Safety checks: each returns (name, ok, detail)
# ======================================================================
def check_resources(cfg):
    problems = []
    du = shutil.disk_usage("/")
    free = du.free * 100 / du.total
    if free < cfg.min_disk_free_pct:
        problems.append(f"disk free {free:.1f}% < {cfg.min_disk_free_pct}%")
    try:
        load = os.getloadavg()[0] / (os.cpu_count() or 1)
        if load > cfg.max_load_per_cpu:
            problems.append(f"load/cpu {load:.2f} > {cfg.max_load_per_cpu}")
    except OSError:
        pass
    try:
        mem = {k: int(v.split()[0]) for k, v in
               (l.split(":", 1) for l in Path("/proc/meminfo").read_text().splitlines())}
        avail = mem["MemAvailable"] * 100 / mem["MemTotal"]
        if avail < cfg.min_mem_available_pct:
            problems.append(f"memory available {avail:.1f}% < {cfg.min_mem_available_pct}%")
    except (OSError, KeyError, ValueError):
        pass
    return "resources", not problems, "; ".join(problems) or "ok"


def check_services(cfg):
    if not cfg.required_services:
        return "services", True, "none configured"
    down = [s for s in cfg.required_services if not tcp_ok(s)]
    return "services", not down, f"not reachable: {down}" if down else "ok"


def listening_ports():
    ports = set()
    for f in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = Path(f).read_text().splitlines()[1:]
        except OSError:
            continue
        for ln in lines:
            p = ln.split()
            if len(p) > 3 and p[3] == "0A":                  # LISTEN
                ports.add(int(p[1].rsplit(":", 1)[1], 16))
    return ports


def check_listeners(cfg):
    if cfg.allowed_listen_ports is None:
        return "listeners", True, "no allowlist configured"
    extra = sorted(listening_ports() - set(cfg.allowed_listen_ports))
    return "listeners", not extra, f"unexpected listening ports: {extra}" if extra else "ok"


def check_uid0(cfg):
    if not cfg.check_uid0:
        return "uid0", True, "disabled"
    extra = []
    try:
        for ln in Path("/etc/passwd").read_text().splitlines():
            parts = ln.split(":")
            if len(parts) > 3 and parts[2] == "0" and parts[0] != "root":
                extra.append(parts[0])
    except OSError as e:
        return "uid0", False, f"cannot read /etc/passwd: {e}"
    return "uid0", not extra, f"extra UID-0 accounts: {extra}" if extra else "ok"


def sha256_file(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except FileNotFoundError:
        return None
    except OSError as e:
        return f"error:{e.errno}"


def baseline_path(cfg):
    return Path(cfg.state_dir) / "baseline.json"


def init_baseline(cfg):
    Path(cfg.state_dir).mkdir(parents=True, exist_ok=True)
    data = {p: sha256_file(p) for p in cfg.integrity_paths}
    baseline_path(cfg).write_text(json.dumps(data, indent=2))
    log.info("baseline written to %s (%d files)", baseline_path(cfg), len(data))


def check_integrity(cfg):
    if not cfg.integrity_paths:
        return "integrity", True, "no paths configured"
    try:
        base = json.loads(baseline_path(cfg).read_text())
    except (OSError, ValueError):
        return "integrity", False, "no baseline (run --init-baseline while the host is known-good)"
    changed = [p for p in sorted(set(base) | set(cfg.integrity_paths))
               if sha256_file(p) != base.get(p, "missing-from-baseline")]
    return "integrity", not changed, f"changed since baseline: {changed}" if changed else "ok"


CHECKS = (check_resources, check_services, check_listeners, check_uid0, check_integrity)


def run_checks(cfg):
    results = []
    for fn in CHECKS:
        try:
            results.append(fn(cfg))
        except Exception as e:                              # a crashing check counts as a failure
            results.append((fn.__name__, False, f"check crashed: {e}"))
    return results


def checks_stable(cfg, stop):
    for i in range(cfg.stable_passes):
        results = run_checks(cfg)
        for name, ok, detail in results:
            log.info("check %-10s %s  %s", name, "PASS" if ok else "FAIL", detail)
        bad = [r for r in results if not r[1]]
        if bad:
            return False, bad
        if i < cfg.stable_passes - 1 and stop.wait(cfg.check_interval):
            return False, [("interrupted", False, "stopped")]
    return True, []


# ======================================================================
# Restore / verify / re-isolate
# ======================================================================
def direct_unblock():
    """Fallback when kill_switch.sh is unavailable: remove its nft table / iptables chains."""
    if shutil.which("nft"):
        run(["nft", "delete", "table", "inet", "killswitch"])
    for binary in ("iptables", "ip6tables"):
        if not shutil.which(binary):
            continue
        for chain, target in (("INPUT", "KS_IN"), ("OUTPUT", "KS_OUT"), ("FORWARD", "KS_FWD")):
            while run([binary, "-w", "-D", chain, "-j", target])[0] == 0:
                pass
            run([binary, "-w", "-F", target])
            run([binary, "-w", "-X", target])


def has_default_route():
    try:
        for ln in Path("/proc/net/route").read_text().splitlines()[1:]:
            p = ln.split()
            if len(p) > 3 and p[1] == "00000000" and int(p[3], 16) & 2:
                return True
    except OSError:
        pass
    return False


def restore_network(cfg):
    ok = False
    if Path(cfg.kill_switch).exists():
        rc, out = run(ks_cmd(cfg, "off"))
        log.info("kill_switch off -> rc=%s %s", rc, out)
        ok = rc == 0
    if not ok:
        log.warning("using direct firewall cleanup")
        direct_unblock()
    for ifc in cfg.interfaces:
        rc, out = run(["ip", "link", "set", "dev", ifc, "up"])
        log.info("ip link set %s up -> rc=%s %s", ifc, rc, out)
    return not rules_active(cfg)


def wait_for_link(cfg, stop, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline and not stop.is_set():
        if not ifaces_down(cfg) and has_default_route():
            return True
        stop.wait(1)
    return not ifaces_down(cfg) and has_default_route()


def verify_internet(cfg, stop):
    deadline = time.time() + cfg.verify_timeout
    while time.time() < deadline and not stop.is_set():
        try:
            socket.getaddrinfo(cfg.dns_probe_host, 443)
            dns = True
        except OSError:
            dns = False
        if dns and any(tcp_ok(t) for t in cfg.probe_targets):
            return True
        stop.wait(3)
    return False


def reisolate(cfg):
    if Path(cfg.kill_switch).exists():
        rc, out = run(ks_cmd(cfg, "on", "--force"))
        log.critical("RE-ISOLATED via kill_switch.sh (rc=%s) %s", rc, out)
    elif cfg.interfaces:
        for ifc in cfg.interfaces:
            run(["ip", "link", "set", "dev", ifc, "down"])
        log.critical("RE-ISOLATED by bringing interfaces down: %s", cfg.interfaces)
    else:
        log.critical("checks failed after reconnect but there is no way to re-isolate "
                     "(set kill_switch or interfaces)")


# ======================================================================
# One cycle
# ======================================================================
def cycle(cfg, stop, dry_run=False):
    if not is_isolated(cfg):
        log.info("not isolated; nothing to do")
        return "not_isolated"
    if Path(cfg.hold_file).exists():
        log.warning("HOLD file present (%s); staying isolated", cfg.hold_file)
        return "hold"
    state = State(cfg.state_dir)
    if len(state.recent()) >= cfg.max_reconnects_per_hour:
        log.error("%d reconnects in the last hour; staying isolated (flapping guard)",
                  len(state.recent()))
        return "rate_limited"

    ok, bad = checks_stable(cfg, stop)
    if not ok:
        log.warning("environment NOT safe: %s; staying isolated", [b[0] for b in bad])
        return "unsafe"
    if dry_run:
        log.info("dry-run: all checks passed; would reconnect now")
        return "dry_run_ok"

    log.info("environment safe; restoring network")
    state.record()
    if not restore_network(cfg):
        log.error("firewall rules still active after restore attempt")
        return "restore_failed"
    if not wait_for_link(cfg, stop):
        log.warning("link/default route not ready")
    if not verify_internet(cfg, stop):
        log.error("rules lifted but internet is not reachable (network problem, not a security one)")
        return "no_internet"

    log.info("internet reachable; re-checking in %ss", cfg.post_check_delay)
    stop.wait(cfg.post_check_delay)
    bad = [r for r in run_checks(cfg) if not r[1]]
    if bad:
        log.critical("post-reconnect checks FAILED: %s", bad)
        reisolate(cfg)
        return "reisolated"
    log.info("RECONNECTED and verified")
    return "reconnected"


EXIT = {"reconnected": 0, "not_isolated": 0, "dry_run_ok": 0, "reisolated": 3}


# ======================================================================
# main
# ======================================================================
def setup_logging():
    fmt = logging.Formatter("%(asctime)s reconnect[%(process)d] %(levelname)s %(message)s")
    h = logging.StreamHandler()
    h.setFormatter(fmt)
    log.addHandler(h)
    log.setLevel(logging.INFO)
    if os.path.exists("/dev/log"):
        try:
            s = logging.handlers.SysLogHandler(address="/dev/log")
            s.setFormatter(logging.Formatter("reconnect: %(levelname)s %(message)s"))
            log.addHandler(s)
        except OSError:
            pass


def acquire_lock(cfg):
    for d in ("/run", cfg.state_dir, tempfile.gettempdir()):
        try:
            Path(d).mkdir(parents=True, exist_ok=True)
            fh = open(Path(d) / "reconnect.lock", "w")
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fh
        except BlockingIOError:
            raise SystemExit("another reconnect.py instance is running")
        except OSError:
            continue
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config")
    ap.add_argument("--watch", action="store_true", help="run forever")
    ap.add_argument("--once", action="store_true", help="single cycle (default)")
    ap.add_argument("--dry-run", action="store_true", help="checks only; change nothing")
    ap.add_argument("--init-baseline", action="store_true")
    ap.add_argument("--kill-switch")
    ap.add_argument("--iface", action="append", default=[])
    ap.add_argument("--stable-passes", type=int)
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    if args.kill_switch:
        cfg.kill_switch = args.kill_switch
    if args.iface:
        cfg.interfaces += args.iface
    if args.stable_passes:
        cfg.stable_passes = args.stable_passes

    if args.init_baseline:
        init_baseline(cfg)
        return 0
    if not args.dry_run and os.geteuid() != 0:
        raise SystemExit("must run as root (or use --dry-run)")

    _lock = acquire_lock(cfg)                                # keep reference alive
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    if not args.watch:
        return EXIT.get(cycle(cfg, stop, args.dry_run), 1)

    log.info("watching (poll every %ss)", cfg.check_interval)
    while not stop.is_set():
        try:
            cycle(cfg, stop, args.dry_run)
        except Exception:
            log.exception("cycle crashed; will retry")
        stop.wait(cfg.check_interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
