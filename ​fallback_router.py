#!/usr/bin/env python3
"""
fallback_router.py - route requests to the primary server, and transparently
fall back to a local cache service when the network is down.

Guarantees
----------
* request() NEVER raises. Every call returns a RouterResponse, and
  `resp.source` says where it came from: "primary", "cache", "queued", "none".
* Fast failover: a circuit breaker stops hitting a dead network, so callers
  don't wait on a timeout for every request while offline.
* Automatic recovery: after `recovery_timeout` the breaker lets one trial
  request through; if it succeeds, traffic returns to the primary.
* Stale-if-error: expired cache entries are still served when the primary is
  down (flagged with resp.stale = True).
* Write-through: every successful primary GET refreshes the cache.

Cache backends (tried in order, all failures are swallowed)
-----------------------------------------------------------
* HttpCache   - your local cache service. Assumed contract:
                   GET {base}/cache/{key} -> 200 JSON entry | 404
                   PUT {base}/cache/{key}    (JSON entry body)
* SQLiteCache - built-in file-based cache; works with no extra service.

Usage
-----
    router = FallbackRouter(
        "https://api.example.com",
        caches=[HttpCache("http://localhost:7070"), SQLiteCache("cache.db")],
    )
    resp = router.get("/users/42")
    print(resp.status, resp.source, resp.stale, resp.json())

Writes while offline (POST/PUT/PATCH/DELETE) can't be answered from a cache.
Pass `on_offline_write=fn(method, path, body, headers)` (for example, enqueue
into offline_queue.py) and the router returns 202 with source="queued".
Standard library only.
"""

import base64
import hashlib
import http.client
import json
import logging
import socket
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import closing
from dataclasses import dataclass, field

log = logging.getLogger("fallback_router")

NETWORK_ERRORS = (urllib.error.URLError, socket.timeout, ConnectionError,
                  http.client.HTTPException, OSError)


# ======================================================================
# Response object (what callers always get back)
# ======================================================================
@dataclass
class RouterResponse:
    status: int
    body: bytes = b""
    headers: dict = field(default_factory=dict)
    source: str = "primary"      # primary | cache | queued | none
    stale: bool = False
    error: str = None            # why the primary wasn't used, if it wasn't

    @property
    def ok(self):
        return 200 <= self.status < 300

    def text(self):
        return self.body.decode("utf-8", errors="replace")

    def json(self, default=None):
        try:
            return json.loads(self.body)
        except (ValueError, TypeError):
            return default


# ======================================================================
# Circuit breaker
# ======================================================================
class CircuitBreaker:
    CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"

    def __init__(self, failure_threshold=3, recovery_timeout=15.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = self.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._lock = threading.Lock()

    def allow(self):
        with self._lock:
            if self.state == self.CLOSED:
                return True
            if (self.state == self.OPEN
                    and time.monotonic() - self._opened_at >= self.recovery_timeout):
                self.state = self.HALF_OPEN      # exactly one trial request
                return True
            return False

    def record_success(self):
        with self._lock:
            if self.state != self.CLOSED:
                log.info("primary is back, circuit closed")
            self.state, self._failures = self.CLOSED, 0

    def record_failure(self):
        with self._lock:
            self._failures += 1
            if self.state == self.HALF_OPEN or self._failures >= self.failure_threshold:
                if self.state != self.OPEN:
                    log.warning("primary unavailable, circuit open")
                self.state = self.OPEN
                self._opened_at = time.monotonic()


# ======================================================================
# Cache backends
# ======================================================================
class SQLiteCache:
    def __init__(self, path="fallback_cache.db"):
        self.path = path
        with closing(self._conn()) as c, c:
            c.execute("CREATE TABLE IF NOT EXISTS cache "
                      "(key TEXT PRIMARY KEY, entry TEXT NOT NULL)")

    def _conn(self):
        return sqlite3.connect(self.path, timeout=10)

    def get(self, key):
        with closing(self._conn()) as c:
            row = c.execute("SELECT entry FROM cache WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key, entry):
        with closing(self._conn()) as c, c:
            c.execute("INSERT INTO cache (key, entry) VALUES (?, ?) "
                      "ON CONFLICT(key) DO UPDATE SET entry=excluded.entry",
                      (key, json.dumps(entry)))


class HttpCache:
    """Client for a local cache service (see module docstring for the contract)."""

    def __init__(self, base_url, timeout=1.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get(self, key):
        try:
            with urllib.request.urlopen(f"{self.base_url}/cache/{key}",
                                        timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise

    def set(self, key, entry):
        req = urllib.request.Request(
            f"{self.base_url}/cache/{key}", data=json.dumps(entry).encode(),
            headers={"Content-Type": "application/json"}, method="PUT")
        with urllib.request.urlopen(req, timeout=self.timeout):
            pass


# ======================================================================
# Router
# ======================================================================
class FallbackRouter:
    def __init__(self, primary_url, caches=None, timeout=3.0, cache_ttl=300,
                 breaker=None, on_offline_write=None):
        self.primary_url = primary_url.rstrip("/")
        self.caches = caches if caches is not None else [SQLiteCache()]
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        self.breaker = breaker or CircuitBreaker()
        self.on_offline_write = on_offline_write

    # ---- convenience verbs ----
    def get(self, path, headers=None):
        return self.request("GET", path, headers=headers)

    def post(self, path, body=None, headers=None):
        return self.request("POST", path, body, headers)

    def put(self, path, body=None, headers=None):
        return self.request("PUT", path, body, headers)

    def delete(self, path, headers=None):
        return self.request("DELETE", path, headers=headers)

    # ---- main entry point: never raises ----
    def request(self, method, path, body=None, headers=None):
        try:
            return self._route(method.upper(), path, body, headers or {})
        except Exception as e:                      # last-resort safety net
            log.exception("unexpected router error")
            return RouterResponse(503, b"", {}, "none", error=f"router error: {e}")

    def _route(self, method, path, body, headers):
        key = self._key(method, path)
        primary_resp, error = None, None

        if self.breaker.allow():
            primary_resp, error = self._call_primary(method, path, body, headers)
            if primary_resp is not None and primary_resp.status < 500:
                self.breaker.record_success()       # 2xx-4xx are real server answers
                if method == "GET" and primary_resp.ok:
                    self._cache_put(key, primary_resp)
                return primary_resp
            self.breaker.record_failure()
        else:
            error = "circuit open (primary marked down)"

        return self._fallback(method, path, body, headers, key, primary_resp, error)

    # ---- primary ----
    def _call_primary(self, method, path, body, headers):
        data = None
        hdrs = dict(headers)
        if body is not None:
            if isinstance(body, (dict, list)):
                data = json.dumps(body).encode()
                hdrs.setdefault("Content-Type", "application/json")
            else:
                data = body.encode() if isinstance(body, str) else body
        req = urllib.request.Request(self.primary_url + path, data=data,
                                     headers=hdrs, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return RouterResponse(r.status, r.read(), dict(r.headers), "primary"), None
        except urllib.error.HTTPError as e:         # server answered with an error code
            try:
                payload = e.read()
            except Exception:
                payload = b""
            return RouterResponse(e.code, payload, dict(e.headers or {}), "primary"), f"HTTP {e.code}"
        except NETWORK_ERRORS as e:
            return None, str(e)

    # ---- fallback ----
    def _fallback(self, method, path, body, headers, key, primary_resp, error):
        if method == "GET":
            entry = self._cache_get(key)
            if entry:
                age = time.time() - entry.get("stored_at", 0)
                hdrs = dict(entry.get("headers", {}))
                hdrs["X-Served-From"] = "cache"
                return RouterResponse(entry["status"], base64.b64decode(entry["body"]),
                                      hdrs, "cache", stale=age > self.cache_ttl, error=error)
            if primary_resp is not None:            # cache miss: return the real 5xx
                primary_resp.error = error
                return primary_resp
            return self._unavailable("no cached copy available", error)

        if self.on_offline_write:                   # POST/PUT/PATCH/DELETE
            try:
                self.on_offline_write(method, path, body, headers)
                return RouterResponse(202, b'{"queued": true}',
                                      {"Content-Type": "application/json"},
                                      "queued", error=error)
            except Exception as e:
                log.warning("offline write hook failed: %s", e)
        if primary_resp is not None:
            primary_resp.error = error
            return primary_resp
        return self._unavailable("write could not be delivered", error)

    @staticmethod
    def _unavailable(message, error):
        payload = json.dumps({"error": "service_unavailable", "detail": message}).encode()
        return RouterResponse(503, payload, {"Content-Type": "application/json"},
                              "none", error=error)

    # ---- cache helpers (every failure is swallowed) ----
    @staticmethod
    def _key(method, path):
        return hashlib.sha256(f"{method} {path}".encode()).hexdigest()

    def _cache_put(self, key, resp):
        entry = {"status": resp.status, "headers": resp.headers,
                 "body": base64.b64encode(resp.body).decode(), "stored_at": time.time()}
        for cache in self.caches:
            try:
                cache.set(key, entry)
            except Exception as e:
                log.debug("cache write failed (%s): %s", type(cache).__name__, e)

    def _cache_get(self, key):
        for cache in self.caches:
            try:
                entry = cache.get(key)
                if entry:
                    return entry
            except Exception as e:
                log.debug("cache read failed (%s): %s", type(cache).__name__, e)
        return None


# ======================================================================
# CLI demo:  python fallback_router.py https://api.example.com /users/42
# ======================================================================
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(0)
    logging.basicConfig(level=logging.INFO)
    r = FallbackRouter(sys.argv[1]).get(sys.argv[2])
    print(f"status={r.status} source={r.source} stale={r.stale} error={r.error}")
    print(r.text()[:500])
