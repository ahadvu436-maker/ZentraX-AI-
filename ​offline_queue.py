#!/usr/bin/env python3
"""
offline_queue.py - cache HTTP requests locally while offline, replay when back online.

- Persistent: queue lives in SQLite, so it survives crashes and restarts.
- Ordered: requests are replayed FIFO.
- Safe retries: each request carries an Idempotency-Key header.
- Zero dependencies: standard library only.

Usage:
    q = OfflineQueue("queue.db")
    q.start()                                   # background flusher
    q.send("POST", "https://api.example.com/events", {"type": "click"})
    ...
    q.stop()

CLI demo:
    python offline_queue.py send POST https://httpbin.org/post '{"hello": "world"}'
    python offline_queue.py flush
    python offline_queue.py status
"""

import json
import socket
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from contextlib import closing
from urllib.parse import urlparse

RETRYABLE_HTTP = {408, 425, 429, 500, 502, 503, 504}


class OfflineQueue:
    def __init__(self, db_path="offline_queue.db", timeout=10, max_attempts=8,
                 flush_interval=5, max_backoff=300):
        self.db_path = db_path
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.flush_interval = flush_interval
        self.max_backoff = max_backoff
        self._lock = threading.Lock()          # serialises flushes
        self._stop = threading.Event()
        self._thread = None
        self._init_db()

    # ---------- storage ----------
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with closing(self._conn()) as c, c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL UNIQUE,
                    method TEXT NOT NULL,
                    url TEXT NOT NULL,
                    headers TEXT NOT NULL,
                    body TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'pending',   -- pending | failed
                    last_error TEXT,
                    created_at REAL NOT NULL
                )""")

    def enqueue(self, method, url, body=None, headers=None):
        key = str(uuid.uuid4())
        payload = json.dumps(body) if body is not None else None
        with closing(self._conn()) as c, c:
            c.execute(
                "INSERT INTO queue (key, method, url, headers, body, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (key, method.upper(), url, json.dumps(headers or {}), payload, time.time()),
            )
        return key

    def pending_count(self):
        with closing(self._conn()) as c:
            return c.execute(
                "SELECT COUNT(*) FROM queue WHERE status='pending'").fetchone()[0]

    # ---------- network ----------
    @staticmethod
    def is_online(host="1.1.1.1", port=53, timeout=3):
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    def _http(self, row):
        headers = {"Content-Type": "application/json", **json.loads(row["headers"])}
        headers["Idempotency-Key"] = row["key"]
        data = row["body"].encode() if row["body"] else None
        req = urllib.request.Request(row["url"], data=data, headers=headers,
                                     method=row["method"])
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return resp.status

    # ---------- public API ----------
    def send(self, method, url, body=None, headers=None):
        """Try now; if the network is down, queue it. Returns 'sent' or 'queued'."""
        key = self.enqueue(method, url, body, headers)
        self.flush()
        with closing(self._conn()) as c:
            still_there = c.execute(
                "SELECT 1 FROM queue WHERE key=?", (key,)).fetchone()
        return "queued" if still_there else "sent"

    def flush(self):
        """Replay pending requests in order. Stops at the first network failure."""
        if not self._lock.acquire(blocking=False):
            return 0
        sent = 0
        try:
            with closing(self._conn()) as c:
                rows = c.execute(
                    "SELECT * FROM queue WHERE status='pending' ORDER BY id").fetchall()
            for row in rows:
                outcome, err = self._attempt(row)
                with closing(self._conn()) as c, c:
                    if outcome == "ok":
                        c.execute("DELETE FROM queue WHERE id=?", (row["id"],))
                        sent += 1
                    elif outcome == "offline":
                        break                        # keep order; try later
                    else:
                        attempts = row["attempts"] + 1
                        status = ("failed" if outcome == "drop"
                                  or attempts >= self.max_attempts else "pending")
                        c.execute(
                            "UPDATE queue SET attempts=?, status=?, last_error=? WHERE id=?",
                            (attempts, status, err, row["id"]))
                        if status == "pending":
                            break                    # server trouble; back off
        finally:
            self._lock.release()
        return sent

    def _attempt(self, row):
        try:
            self._http(row)
            return "ok", None
        except urllib.error.HTTPError as e:
            if e.code in RETRYABLE_HTTP:
                return "retry", f"HTTP {e.code}"
            return "drop", f"HTTP {e.code}"          # 4xx: retrying won't help
        except (urllib.error.URLError, socket.timeout, ConnectionError, OSError) as e:
            return "offline", str(e)

    # ---------- background worker ----------
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        delay = self.flush_interval
        while not self._stop.is_set():
            if self.pending_count() and self.is_online():
                self.flush()
                delay = (self.flush_interval if self.pending_count() == 0
                         else min(delay * 2, self.max_backoff))
            else:
                delay = self.flush_interval
            self._stop.wait(delay)


# ---------- CLI ----------
def main(argv):
    q = OfflineQueue()
    cmd = argv[1] if len(argv) > 1 else "status"
    if cmd == "send" and len(argv) >= 4:
        body = json.loads(argv[4]) if len(argv) > 4 else None
        print(q.send(argv[2], argv[3], body))
    elif cmd == "flush":
        print(f"sent {q.flush()}, pending {q.pending_count()}")
    elif cmd == "status":
        print(f"pending: {q.pending_count()}")
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv)
