"""SQLite database tool with read-only-by-default access.

Actions: ``query`` (SELECT only), ``execute`` (writes; needs ``db.write`` AND ``read_only: False``
in config), ``list_tables``, ``describe``.  Safety measures: parameterised statements, SQLite
authoriser (read queries may only SELECT/READ; writes may not ATTACH/PRAGMA), ``mode=ro`` for file
databases, row caps, and a progress handler that aborts queries past the call deadline.

Config keys: ``databases`` (``{alias: path}``; default ``{"default": ":memory:"}``),
``read_only`` (default True), ``max_rows`` (default 200).
"""
from __future__ import annotations

import re
import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Mapping, Optional, Set

from ..tool_core import (
    BaseTool, ToolCategory, ToolContext, ToolError, ToolExecutionError, ToolParameter,
    ToolPermissionError, ToolSpec, ToolTimeoutError, ToolValidationError,
)

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_WRITE_ACTIONS = {"execute"}
_READ_OK = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION,
            getattr(sqlite3, "SQLITE_RECURSIVE", 33)}
_WRITE_DENIED = {sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH, sqlite3.SQLITE_PRAGMA}


class DatabaseTool(BaseTool):
    """Query and (optionally) modify SQLite databases."""

    spec = ToolSpec(
        name="database",
        description=("Run SQL against a configured SQLite database. Actions: 'query' (read-only SELECT), "
                     "'execute' (INSERT/UPDATE/DELETE/DDL, if enabled), 'list_tables', 'describe' (table "
                     "columns). Use '?' placeholders with 'params' for values."),
        parameters=(
            ToolParameter("action", "string", "Operation.", enum=("query", "execute", "list_tables", "describe")),
            ToolParameter("sql", "string", "SQL statement (query/execute).", required=False, default=None,
                          max_length=20_000),
            ToolParameter("params", "array", "Positional values for '?' placeholders.", required=False,
                          default=None),
            ToolParameter("table", "string", "Table name (describe).", required=False, default=None,
                          max_length=128),
            ToolParameter("database", "string", "Database alias.", required=False, default="default",
                          max_length=64),
            ToolParameter("max_rows", "integer", "Maximum rows to return.", required=False, default=100,
                          minimum=1, maximum=5000),
        ),
        category=ToolCategory.DATABASE, tags=("sql", "sqlite"),
        required_permissions=frozenset({"db.read"}), timeout=30.0,
    )

    def required_permissions_for(self, arguments: Mapping[str, Any]) -> Set[str]:
        return {"db.write"} if arguments.get("action") in _WRITE_ACTIONS else {"db.read"}

    def setup(self) -> None:
        self._dbs: Dict[str, str] = dict(self.config.get("databases") or {"default": ":memory:"})
        self._read_only = bool(self.config.get("read_only", True))
        self._max_rows = int(self.config.get("max_rows", 200))
        self._mem: Dict[str, sqlite3.Connection] = {}
        self._mem_lock = threading.RLock()

    def teardown(self) -> None:
        for conn in self._mem.values():
            conn.close()
        self._mem.clear()

    # -- connections ---------------------------------------------------------- #
    @contextmanager
    def _connection(self, alias: str, readonly: bool) -> Iterator[sqlite3.Connection]:
        if alias not in self._dbs:
            raise ToolValidationError(f"unknown database '{alias}'; available: {sorted(self._dbs)}")
        path = self._dbs[alias]
        if path == ":memory:":
            with self._mem_lock:  # one shared connection, serialised
                conn = self._mem.get(alias) or sqlite3.connect(":memory:", check_same_thread=False)
                self._mem[alias] = conn
                try:
                    yield conn
                finally:
                    conn.set_authorizer(None)
                    conn.set_progress_handler(None, 0)
            return
        uri = f"file:{path}?mode={'ro' if readonly else 'rwc'}"
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False, timeout=5)
        try:
            yield conn
        finally:
            conn.close()

    @staticmethod
    def _guard(conn: sqlite3.Connection, context: ToolContext, *, read_only: bool) -> None:
        def authorizer(action: int, *_: Any) -> int:
            if read_only:
                return sqlite3.SQLITE_OK if action in _READ_OK else sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_DENY if action in _WRITE_DENIED else sqlite3.SQLITE_OK

        def progress() -> int:
            left = context.remaining()
            return 1 if context.cancel_event.is_set() or (left is not None and left <= 0) else 0

        conn.set_authorizer(authorizer)
        conn.set_progress_handler(progress, 10_000)

    # -- actions ----------------------------------------------------------------- #
    def run(self, context: ToolContext, action: str, sql: Optional[str] = None, params: Optional[List[Any]] = None,
            table: Optional[str] = None, database: str = "default", max_rows: int = 100, **_: Any) -> Any:
        max_rows = min(max_rows, self._max_rows)
        try:
            if action == "list_tables":
                with self._connection(database, True) as conn:
                    return self._fetch(conn.execute(
                        "SELECT name, type FROM sqlite_master WHERE type IN ('table','view') "
                        "AND name NOT LIKE 'sqlite_%' ORDER BY name"), max_rows)
            if action == "describe":
                if not table or not _IDENT.match(table):
                    raise ToolValidationError("'table' must be a plain identifier")
                with self._connection(database, True) as conn:
                    out = self._fetch(conn.execute(f'PRAGMA table_info("{table}")'), max_rows)
                if not out["rows"]:
                    raise ToolExecutionError(f"table '{table}' not found")
                out["columns"] = ["cid", "name", "type", "notnull", "default", "pk"]
                return out
            if not sql or not sql.strip():
                raise ToolValidationError("'sql' is required")
            if action == "query":
                with self._connection(database, True) as conn:
                    self._guard(conn, context, read_only=True)
                    return self._fetch(conn.execute(sql, params or []), max_rows)
            if action == "execute":
                if self._read_only:
                    raise ToolPermissionError("database tool is configured read-only")
                with self._connection(database, False) as conn:
                    self._guard(conn, context, read_only=False)
                    try:
                        cur = conn.execute(sql, params or [])
                        conn.commit()
                    except BaseException:
                        conn.rollback()
                        raise
                    return {"rowcount": cur.rowcount, "lastrowid": cur.lastrowid}
            raise ToolValidationError(f"unknown action '{action}'")  # pragma: no cover
        except ToolError:
            raise
        except sqlite3.OperationalError as exc:
            if "interrupted" in str(exc).lower():
                raise ToolTimeoutError("query aborted: deadline exceeded or cancelled") from None
            raise ToolExecutionError(f"SQL error: {exc}") from None
        except sqlite3.Error as exc:
            raise ToolExecutionError(f"SQL error: {exc}") from None

    @staticmethod
    def _fetch(cursor: sqlite3.Cursor, max_rows: int) -> Dict[str, Any]:
        columns = [d[0] for d in cursor.description] if cursor.description else []
        raw = cursor.fetchmany(max_rows + 1)
        rows = [[(f"<{len(v)} bytes>" if isinstance(v, (bytes, bytearray)) else v) for v in row]
                for row in raw[:max_rows]]
        return {"columns": columns, "rows": rows, "row_count": len(rows), "truncated": len(raw) > max_rows}
