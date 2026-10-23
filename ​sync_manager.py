#!/usr/bin/env python3
"""
sync_manager.py - automatically sync and merge local data with the main server
as soon as the server comes online.

How it works
------------
* LocalStore keeps records in SQLite. Every local edit marks the record "dirty".
* SyncManager watches the server's health endpoint. On the offline -> online
  transition (or whenever local edits happen while online) it runs a sync:
      1. PULL  server changes since the last cursor, merging into local data
      2. PUSH  dirty local records in batches; the server answers per record
               with "ok" or "conflict" (and its current copy)
      3. On conflict, merge locally and retry on the next loop
* Merge = three-way, field-level (base = last synced copy):
      only local changed  -> keep local
      only server changed -> keep server
      both changed        -> newest updated_at wins for that field
  A delete vs. an edit is decided by the newest updated_at.

Server contract (adapt to your API)
-----------------------------------
GET  {base}/health                 -> 200 when up
GET  {base}/sync/pull?since=CURSOR -> {"changes": [{"id","data","version",
                                        "updated_at","deleted"}], "cursor": "..."}
POST {base}/sync/push              body {"changes": [{"id","data","base_version",
                                        "updated_at","deleted"}]}
                                   -> {"results": [
                                        {"id","status":"ok","version":N} |
                                        {"id","status":"conflict","server":{...}} ]}

Standard library only.

Usage
-----
    store = LocalStore("local.db")
    sync = SyncManager(store, "https://api.example.com", token="...")
    sync.start()
    store.put("note-1", {"title": "Hello"})   # synced automatically when online
    sync.stop()

CLI:  python sync_manager.py put ID '{"k": "v"}' | delete ID | sync | status
"""

import json
import os
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import closing

_MISSING = object()
BATCH_SIZE = 100
MAX_ROUNDS = 5          # push/merge retries per sync run


class NetworkError(Exception):
    """Server unreachable / timed out."""


class ServerError(Exception):
    """Server answered with an error status."""


# ======================================================================
# Local storage
# ======================================================================
class LocalStore:
    def __init__(self, path="local.db", on_change=None):
        self.path = path
        self.on_change = on_change      # called after local edits (used to trigger sync)
        with closing(self._conn()) as c, c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS records (
                    id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    base TEXT,                       -- last synced copy (for 3-way merge)
                    version INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL,
                    deleted INTEGER NOT NULL DEFAULT 0,
                    dirty INTEGER NOT NULL DEFAULT 0
                )""")
            c.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)")

    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    # ---- public data API ----
    def put(self, rec_id, data):
        now = time.time()
        with closing(self._conn()) as c, c:
            c.execute("""
                INSERT INTO records (id, data, updated_at, dirty) VALUES (?, ?, ?, 1)
                ON CONFLICT(id) DO UPDATE SET
                    data=excluded.data, updated_at=excluded.updated_at,
                    deleted=0, dirty=1""", (rec_id, json.dumps(data), now))
        self._notify()

    def delete(self, rec_id):
        with closing(self._conn()) as c, c:
            c.execute("UPDATE records SET deleted=1, dirty=1, updated_at=? WHERE id=?",
                      (time.time(), rec_id))
        self._notify()

    def get(self, rec_id):
        with closing(self._conn()) as c:
            row = c.execute("SELECT data FROM records WHERE id=? AND deleted=0",
                            (rec_id,)).fetchone()
        return json.loads(row["data"]) if row else None

    def dirty_count(self):
        with closing(self._conn()) as c:
            return c.execute("SELECT COUNT(*) FROM records WHERE dirty=1").fetchone()[0]

    # ---- sync internals ----
    def dirty_batch(self):
        with closing(self._conn()) as c:
            return c.execute("SELECT * FROM records WHERE dirty=1 ORDER BY updated_at "
                             "LIMIT ?", (BATCH_SIZE,)).fetchall()

    def row(self, rec_id):
        with closing(self._conn()) as c:
            return c.execute("SELECT * FROM records WHERE id=?", (rec_id,)).fetchone()

    def get_meta(self, key, default=None):
        with closing(self._conn()) as c:
            r = c.execute("SELECT v FROM meta WHERE k=?", (key,)).fetchone()
        return r["v"] if r else default

    def set_meta(self, key, value):
        with closing(self._conn()) as c, c:
            c.execute("INSERT INTO meta (k, v) VALUES (?, ?) "
                      "ON CONFLICT(k) DO UPDATE SET v=excluded.v", (key, value))

    def _notify(self):
        if self.on_change:
            self.on_change()


# ======================================================================
# Merge logic
# ======================================================================
def merge_fields(base, local, server, local_ts, server_ts):
    """Three-way, field-level merge of two dicts against a common base."""
    base, local, server = base or {}, local or {}, server or {}
    out = {}
    for key in set(base) | set(local) | set(server):
        b, l, s = base.get(key, _MISSING), local.get(key, _MISSING), server.get(key, _MISSING)
        if l == s:
            val = l
        elif l == b:
            val = s                              # only the server changed it
        elif s == b:
            val = l                              # only we changed it
        else:
            val = l if local_ts >= server_ts else s   # both changed: newest wins
        if val is not _MISSING:
            out[key] = val
    return out


def resolve(local_row, server):
    """Return (merged_data, deleted) for a local row vs. a server record."""
    local_ts, server_ts = local_row["updated_at"], server["updated_at"]
    if local_row["deleted"] or server.get("deleted"):
        if local_ts >= server_ts:
            return json.loads(local_row["data"]), bool(local_row["deleted"])
        return server["data"], bool(server.get("deleted"))
    base = json.loads(local_row["base"]) if local_row["base"] else {}
    merged = merge_fields(base, json.loads(local_row["data"]), server["data"],
                          local_ts, server_ts)
    return merged, False


# ======================================================================
# Sync manager
# ======================================================================
class SyncManager:
    def __init__(self, store, base_url, token=None, timeout=10,
                 check_interval=5, max_backoff=120):
        self.store = store
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.check_interval = check_interval
        self.max_backoff = max_backoff
        self.online = False
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._sync_lock = threading.Lock()
        self._thread = None
        store.on_change = self.trigger

    # ---- http ----
    def _request(self, method, path, body=None):
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base_url + path, data=data,
                                     headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            raise ServerError(f"HTTP {e.code}") from e
        except (urllib.error.URLError, OSError) as e:      # includes timeouts
            raise NetworkError(str(e)) from e

    def server_is_up(self):
        try:
            self._request("GET", "/health")
            return True
        except (NetworkError, ServerError):
            return False

    # ---- sync steps ----
    def pull(self):
        while True:
            cursor = self.store.get_meta("cursor", "")
            resp = self._request("GET", "/sync/pull?since=" + urllib.parse.quote(cursor))
            changes = resp.get("changes", [])
            for srv in changes:
                self._apply_server_change(srv)
            self.store.set_meta("cursor", resp.get("cursor", cursor))
            if not changes:
                return

    def _apply_server_change(self, srv):
        local = self.store.row(srv["id"])
        with closing(self.store._conn()) as c, c:
            if local is None or not local["dirty"]:
                # nothing to merge: server copy wins
                c.execute("""
                    INSERT INTO records (id, data, base, version, updated_at, deleted, dirty)
                    VALUES (?, ?, ?, ?, ?, ?, 0)
                    ON CONFLICT(id) DO UPDATE SET data=excluded.data, base=excluded.base,
                        version=excluded.version, updated_at=excluded.updated_at,
                        deleted=excluded.deleted, dirty=0""",
                    (srv["id"], json.dumps(srv["data"]), json.dumps(srv["data"]),
                     srv["version"], srv["updated_at"], int(bool(srv.get("deleted")))))
            else:
                data, deleted = resolve(local, srv)
                c.execute("""UPDATE records SET data=?, base=?, version=?, deleted=?, dirty=1
                             WHERE id=?""",
                          (json.dumps(data), json.dumps(srv["data"]), srv["version"],
                           int(deleted), srv["id"]))

    def push(self):
        """Push one batch. Returns True if more dirty records remain."""
        batch = self.store.dirty_batch()
        if not batch:
            return False
        payload = {"changes": [{
            "id": r["id"], "data": json.loads(r["data"]), "base_version": r["version"],
            "updated_at": r["updated_at"], "deleted": bool(r["deleted"]),
        } for r in batch]}
        results = {x["id"]: x for x in self._request("POST", "/sync/push", payload)["results"]}
        sent = {r["id"]: r for r in batch}

        for rec_id, res in results.items():
            row = sent.get(rec_id)
            if row is None:
                continue
            if res["status"] == "ok":
                with closing(self.store._conn()) as c, c:
                    # stay dirty if edited again while the request was in flight
                    c.execute("""UPDATE records SET version=?, base=?,
                                 dirty=CASE WHEN updated_at=? THEN 0 ELSE 1 END
                                 WHERE id=?""",
                              (res["version"], row["data"], row["updated_at"], rec_id))
            elif res["status"] == "conflict":
                self._apply_server_change(res["server"])   # merge; stays dirty

        with closing(self.store._conn()) as c, c:           # purge synced tombstones
            c.execute("DELETE FROM records WHERE deleted=1 AND dirty=0")
        return self.store.dirty_count() > 0

    def sync_once(self):
        """Full pull -> push -> merge cycle. Raises NetworkError if the server drops."""
        if not self._sync_lock.acquire(blocking=False):
            return
        try:
            self.pull()
            for _ in range(MAX_ROUNDS):
                if not self.push():
                    break
                self.pull()          # pick up whatever caused conflicts, then retry
        finally:
            self._sync_lock.release()

    # ---- background watcher ----
    def trigger(self):
        """Wake the worker now (called automatically after local edits)."""
        self._wake.set()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        delay = self.check_interval
        was_online = False
        while not self._stop.is_set():
            up = self.server_is_up()
            self.online = up
            if up:
                # sync on reconnect, and whenever there is local work pending
                if (not was_online) or self.store.dirty_count():
                    try:
                        self.sync_once()
                        delay = self.check_interval
                    except (NetworkError, ServerError) as e:
                        print(f"[sync] failed: {e}", file=sys.stderr)
                        delay = min(delay * 2, self.max_backoff)
            else:
                delay = min(delay * 2, self.max_backoff) if was_online is False \
                    else self.check_interval
            was_online = up
            self._wake.wait(delay)
            self._wake.clear()


# ======================================================================
# CLI
# ======================================================================
def main(argv):
    store = LocalStore(os.environ.get("LOCAL_DB", "local.db"))
    sync = SyncManager(store, os.environ.get("SYNC_URL", "http://localhost:8000"),
                       token=os.environ.get("SYNC_TOKEN"))
    cmd = argv[1] if len(argv) > 1 else "status"
    if cmd == "put" and len(argv) == 4:
        store.put(argv[2], json.loads(argv[3]))
    elif cmd == "delete" and len(argv) == 3:
        store.delete(argv[2])
    elif cmd == "sync":
        try:
            sync.sync_once()
            print("sync complete")
        except (NetworkError, ServerError) as e:
            print(f"sync failed: {e}")
    elif cmd == "status":
        print(f"unsynced records: {store.dirty_count()}")
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv)
