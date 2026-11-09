#!/usr/bin/env python3
"""
user_engagement.py — ZentraX-AI user growth & engagement tracker.

Built for human marketing teams. Uses only the Python standard library and a
local SQLite file, so no setup is required.

Quick start
-----------
    python user_engagement.py demo                       # load sample data + report
    python user_engagement.py add-user u1 --source ads   # register a user
    python user_engagement.py log u1 feature_used --detail "chat"
    python user_engagement.py report --days 30           # print metrics
    python user_engagement.py export-csv logs.csv        # export interaction log

Or import it:
    from user_engagement import EngagementTracker
    t = EngagementTracker()
    t.add_user("u1", source="organic")
    t.log_interaction("u1", "login")
    print(t.report(days=30))
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

DEFAULT_DB = Path(__file__).with_name("engagement.db")

# Event that counts as "activated" (the user's first moment of value).
# Change to match the definition in campaigns_strategy.md (Section 4).
ACTIVATION_EVENT = "first_success"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


class EngagementTracker:
    """Stores users and interaction logs, and computes marketing metrics."""

    def __init__(self, db_path: Path | str = DEFAULT_DB):
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    # ------------------------------------------------------------------ setup
    def _create_tables(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id    TEXT PRIMARY KEY,
                signup_at  TEXT NOT NULL,
                source     TEXT DEFAULT 'unknown',   -- channel/campaign
                country    TEXT DEFAULT NULL
            );
            CREATE TABLE IF NOT EXISTS interactions (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    TEXT NOT NULL,
                event      TEXT NOT NULL,            -- e.g. login, feature_used
                detail     TEXT DEFAULT '',
                ts         TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            );
            CREATE INDEX IF NOT EXISTS idx_int_ts   ON interactions(ts);
            CREATE INDEX IF NOT EXISTS idx_int_user ON interactions(user_id);
            """
        )
        self.conn.commit()

    # ---------------------------------------------------------------- writing
    def add_user(self, user_id: str, source: str = "unknown",
                 country: Optional[str] = None,
                 signup_at: Optional[datetime] = None) -> None:
        """Register a new user (ignored if the user already exists)."""
        self.conn.execute(
            "INSERT OR IGNORE INTO users VALUES (?, ?, ?, ?)",
            (user_id, _iso(signup_at or _now()), source, country),
        )
        self.conn.commit()

    def log_interaction(self, user_id: str, event: str, detail: str = "",
                        ts: Optional[datetime] = None) -> None:
        """Record one interaction event for a user."""
        self.conn.execute(
            "INSERT INTO interactions (user_id, event, detail, ts) VALUES (?, ?, ?, ?)",
            (user_id, event, detail, _iso(ts or _now())),
        )
        self.conn.commit()

    # ---------------------------------------------------------------- metrics
    def _since(self, days: int) -> str:
        return _iso(_now() - timedelta(days=days))

    def new_users(self, days: int = 30) -> int:
        """Signups within the last `days` days."""
        return self.conn.execute(
            "SELECT COUNT(*) FROM users WHERE signup_at >= ?", (self._since(days),)
        ).fetchone()[0]

    def total_users(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    def active_users(self, days: int = 1) -> int:
        """Distinct users with any interaction in the last `days` days."""
        return self.conn.execute(
            "SELECT COUNT(DISTINCT user_id) FROM interactions WHERE ts >= ?",
            (self._since(days),),
        ).fetchone()[0]

    def stickiness(self) -> float:
        """DAU/MAU ratio (higher = users return more often)."""
        mau = self.active_users(30)
        return round(self.active_users(1) / mau, 3) if mau else 0.0

    def engagement_rate(self, days: int = 30) -> float:
        """Share of all users who were active in the period (0–1)."""
        total = self.total_users()
        return round(self.active_users(days) / total, 3) if total else 0.0

    def activation_rate(self) -> float:
        """Share of users who reached the ACTIVATION_EVENT at least once."""
        total = self.total_users()
        if not total:
            return 0.0
        activated = self.conn.execute(
            "SELECT COUNT(DISTINCT user_id) FROM interactions WHERE event = ?",
            (ACTIVATION_EVENT,),
        ).fetchone()[0]
        return round(activated / total, 3)

    def retention(self, day_n: int = 7) -> float:
        """
        Day-N retention: of users who signed up at least N days ago, the share
        with an interaction on or after day N since signup.
        """
        cutoff = self._since(day_n)
        rows = self.conn.execute(
            """
            SELECT u.user_id,
                   EXISTS (
                       SELECT 1 FROM interactions i
                       WHERE i.user_id = u.user_id
                         AND julianday(i.ts) - julianday(u.signup_at) >= ?
                   ) AS returned
            FROM users u WHERE u.signup_at <= ?
            """,
            (day_n, cutoff),
        ).fetchall()
        return round(sum(r["returned"] for r in rows) / len(rows), 3) if rows else 0.0

    def growth_rate(self, days: int = 30) -> float:
        """Signups this period vs. the previous equal period (e.g., 0.25 = +25%)."""
        now = _now()
        cur = self.new_users(days)
        prev = self.conn.execute(
            "SELECT COUNT(*) FROM users WHERE signup_at >= ? AND signup_at < ?",
            (_iso(now - timedelta(days=2 * days)), _iso(now - timedelta(days=days))),
        ).fetchone()[0]
        return round((cur - prev) / prev, 3) if prev else 0.0

    def signups_by_source(self, days: int = 30) -> dict[str, int]:
        """Signups per acquisition channel — shows which campaigns work."""
        rows = self.conn.execute(
            "SELECT source, COUNT(*) c FROM users WHERE signup_at >= ? "
            "GROUP BY source ORDER BY c DESC",
            (self._since(days),),
        ).fetchall()
        return {r["source"]: r["c"] for r in rows}

    def top_events(self, days: int = 30, limit: int = 10) -> dict[str, int]:
        """Most frequent interaction events."""
        rows = self.conn.execute(
            "SELECT event, COUNT(*) c FROM interactions WHERE ts >= ? "
            "GROUP BY event ORDER BY c DESC LIMIT ?",
            (self._since(days), limit),
        ).fetchall()
        return {r["event"]: r["c"] for r in rows}

    # -------------------------------------------------------------- reporting
    def report(self, days: int = 30) -> dict:
        """Return all key metrics as a dictionary."""
        return {
            "period_days": days,
            "total_users": self.total_users(),
            "new_users": self.new_users(days),
            "growth_rate_vs_prev_period": self.growth_rate(days),
            "active_users_period": self.active_users(days),
            "dau": self.active_users(1),
            "mau": self.active_users(30),
            "stickiness_dau_mau": self.stickiness(),
            "engagement_rate": self.engagement_rate(days),
            "activation_rate": self.activation_rate(),
            "retention_d7": self.retention(7),
            "retention_d30": self.retention(30),
            "signups_by_source": self.signups_by_source(days),
            "top_events": self.top_events(days),
        }

    def print_report(self, days: int = 30) -> None:
        """Pretty-print the report for quick reading."""
        r = self.report(days)
        pct = lambda x: f"{x * 100:.1f}%"
        print(f"\n=== ZentraX-AI Engagement Report (last {days} days) ===")
        print(f"Total users:        {r['total_users']}")
        print(f"New users:          {r['new_users']}  (growth {pct(r['growth_rate_vs_prev_period'])})")
        print(f"DAU / MAU:          {r['dau']} / {r['mau']}  (stickiness {pct(r['stickiness_dau_mau'])})")
        print(f"Engagement rate:    {pct(r['engagement_rate'])}")
        print(f"Activation rate:    {pct(r['activation_rate'])}")
        print(f"Retention D7 / D30: {pct(r['retention_d7'])} / {pct(r['retention_d30'])}")
        print("Signups by source:  " + json.dumps(r["signups_by_source"]))
        print("Top events:         " + json.dumps(r["top_events"]) + "\n")

    def export_csv(self, path: str) -> int:
        """Export the full interaction log to CSV. Returns rows written."""
        rows = self.conn.execute(
            "SELECT i.ts, i.user_id, u.source, i.event, i.detail "
            "FROM interactions i JOIN users u USING (user_id) ORDER BY i.ts"
        ).fetchall()
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "user_id", "source", "event", "detail"])
            w.writerows([tuple(r) for r in rows])
        return len(rows)


# ------------------------------------------------------------------ demo data
def load_demo_data(t: EngagementTracker) -> None:
    """Populate sample data so the report can be tried immediately."""
    import random
    random.seed(7)
    sources = ["organic", "ads", "referral", "community"]
    events = ["login", "feature_used", "first_success", "share"]
    for i in range(60):
        signup = _now() - timedelta(days=random.randint(0, 59))
        uid = f"demo_{i}"
        t.add_user(uid, source=random.choice(sources), signup_at=signup)
        for _ in range(random.randint(0, 8)):
            ts = signup + timedelta(days=random.randint(0, 30), hours=random.randint(0, 23))
            if ts <= _now():
                t.log_interaction(uid, random.choice(events), ts=ts)


# ------------------------------------------------------------------------ CLI
def main() -> None:
    p = argparse.ArgumentParser(description="ZentraX-AI engagement tracker")
    p.add_argument("--db", default=str(DEFAULT_DB), help="SQLite file path")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add-user");  a.add_argument("user_id")
    a.add_argument("--source", default="unknown"); a.add_argument("--country")
    l = sub.add_parser("log");       l.add_argument("user_id"); l.add_argument("event")
    l.add_argument("--detail", default="")
    r = sub.add_parser("report");    r.add_argument("--days", type=int, default=30)
    e = sub.add_parser("export-csv"); e.add_argument("path")
    sub.add_parser("demo")

    args = p.parse_args()
    t = EngagementTracker(args.db)
    if args.cmd == "add-user":
        t.add_user(args.user_id, args.source, args.country)
    elif args.cmd == "log":
        t.log_interaction(args.user_id, args.event, args.detail)
    elif args.cmd == "report":
        t.print_report(args.days)
    elif args.cmd == "export-csv":
        print(f"Exported {t.export_csv(args.path)} rows to {args.path}")
    elif args.cmd == "demo":
        load_demo_data(t)
        t.print_report(30)


if __name__ == "__main__":
    main()
