#!/usr/bin/env python3
"""
budget_manager.py — Monthly budget tracker for ZentraX-AI finance team.

Tracks income, operating expenses, and profit margins. Data lives in one JSON
file per month inside ./data/ (e.g. data/2026-11.json), so it is easy to
review, back up, and open in a spreadsheet via the CSV export.

Quick start
-----------
    python budget_manager.py init 2026-11                       # create month
    python budget_manager.py income 2026-11 "Subscriptions" 12000
    python budget_manager.py expense 2026-11 "Cloud hosting" 3200 --category Infrastructure
    python budget_manager.py summary 2026-11
    python budget_manager.py trend                              # all months
    python budget_manager.py export 2026-11 out.csv

Amounts are plain numbers in one currency (set CURRENCY below).
Expense categories may be anything; BUDGET_LIMITS lets you set caps.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import date
from pathlib import Path

DATA_DIR = Path(__file__).with_name("data")
CURRENCY = "USD"

# Optional monthly caps per expense category. Overspend is flagged in summaries.
BUDGET_LIMITS = {
    "Infrastructure": 5000,
    "Marketing": 8000,
    "Payroll": 20000,
    "Legal & Compliance": 2000,
    "Software & Tools": 1500,
    "Other": 1000,
}

MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


# ---------------------------------------------------------------- persistence
def _path(month: str) -> Path:
    if not MONTH_RE.match(month):
        raise SystemExit("Month must look like YYYY-MM, e.g. 2026-11")
    return DATA_DIR / f"{month}.json"


def load_month(month: str) -> dict:
    """Load a month's ledger, creating an empty one if it doesn't exist."""
    path = _path(month)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"month": month, "currency": CURRENCY, "income": [], "expenses": []}


def save_month(data: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _path(data["month"]).write_text(json.dumps(data, indent=2), encoding="utf-8")


# ------------------------------------------------------------------ recording
def add_income(month: str, source: str, amount: float, note: str = "") -> None:
    data = load_month(month)
    data["income"].append({"date": date.today().isoformat(), "source": source,
                           "amount": round(amount, 2), "note": note})
    save_month(data)


def add_expense(month: str, name: str, amount: float,
                category: str = "Other", note: str = "") -> None:
    data = load_month(month)
    data["expenses"].append({"date": date.today().isoformat(), "name": name,
                             "category": category, "amount": round(amount, 2),
                             "note": note})
    save_month(data)


# ------------------------------------------------------------------- analysis
def summarize(month: str) -> dict:
    """Compute totals, profit, margin, and category-level budget status."""
    data = load_month(month)
    income = sum(i["amount"] for i in data["income"])
    expenses = sum(e["amount"] for e in data["expenses"])
    profit = income - expenses
    margin = (profit / income * 100) if income else 0.0

    by_cat: dict[str, float] = {}
    for e in data["expenses"]:
        by_cat[e["category"]] = by_cat.get(e["category"], 0) + e["amount"]

    over_budget = {c: {"spent": v, "limit": BUDGET_LIMITS[c]}
                   for c, v in by_cat.items()
                   if c in BUDGET_LIMITS and v > BUDGET_LIMITS[c]}

    return {"month": month, "income": round(income, 2),
            "expenses": round(expenses, 2), "profit": round(profit, 2),
            "profit_margin_pct": round(margin, 1),
            "expenses_by_category": by_cat, "over_budget": over_budget}


def print_summary(month: str) -> None:
    s = summarize(month)
    print(f"\n=== Budget Summary {s['month']} ({CURRENCY}) ===")
    print(f"Income:        {s['income']:>12,.2f}")
    print(f"Expenses:      {s['expenses']:>12,.2f}")
    print(f"Net profit:    {s['profit']:>12,.2f}")
    print(f"Profit margin: {s['profit_margin_pct']:>11.1f}%")
    print("\nExpenses by category:")
    for cat, amt in sorted(s["expenses_by_category"].items(), key=lambda x: -x[1]):
        limit = BUDGET_LIMITS.get(cat)
        flag = "  ⚠ OVER BUDGET" if cat in s["over_budget"] else ""
        cap = f" / {limit:,.0f}" if limit else ""
        print(f"  {cat:<22}{amt:>10,.2f}{cap}{flag}")
    print()


def print_trend() -> None:
    """Show income, expenses, and margin for every recorded month."""
    if not DATA_DIR.exists():
        print("No data yet.")
        return
    print(f"\n{'Month':<9}{'Income':>12}{'Expenses':>12}{'Profit':>12}{'Margin':>9}")
    for f in sorted(DATA_DIR.glob("*.json")):
        s = summarize(f.stem)
        print(f"{s['month']:<9}{s['income']:>12,.2f}{s['expenses']:>12,.2f}"
              f"{s['profit']:>12,.2f}{s['profit_margin_pct']:>8.1f}%")
    print()


def export_csv(month: str, out: str) -> None:
    """Write income and expenses to one CSV (opens in Excel/Sheets)."""
    data = load_month(month)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["type", "date", "name_or_source", "category", "amount", "note"])
        for i in data["income"]:
            w.writerow(["income", i["date"], i["source"], "", i["amount"], i["note"]])
        for e in data["expenses"]:
            w.writerow(["expense", e["date"], e["name"], e["category"], e["amount"], e["note"]])
    print(f"Exported to {out}")


# ------------------------------------------------------------------------ CLI
def main() -> None:
    p = argparse.ArgumentParser(description="ZentraX-AI monthly budget manager")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init").add_argument("month")
    i = sub.add_parser("income");  i.add_argument("month"); i.add_argument("source")
    i.add_argument("amount", type=float); i.add_argument("--note", default="")
    e = sub.add_parser("expense"); e.add_argument("month"); e.add_argument("name")
    e.add_argument("amount", type=float); e.add_argument("--category", default="Other")
    e.add_argument("--note", default="")
    sub.add_parser("summary").add_argument("month")
    sub.add_parser("trend")
    x = sub.add_parser("export"); x.add_argument("month"); x.add_argument("out")

    a = p.parse_args()
    if a.cmd == "init":
        save_month(load_month(a.month)); print(f"Initialized {a.month}")
    elif a.cmd == "income":
        add_income(a.month, a.source, a.amount, a.note)
    elif a.cmd == "expense":
        add_expense(a.month, a.name, a.amount, a.category, a.note)
    elif a.cmd == "summary":
        print_summary(a.month)
    elif a.cmd == "trend":
        print_trend()
    elif a.cmd == "export":
        export_csv(a.month, a.out)


if __name__ == "__main__":
    main()
