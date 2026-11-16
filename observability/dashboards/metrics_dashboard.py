"""Structured telemetry dashboards.

Two complementary views of the same definition:

1. **Definition -> Grafana**: :meth:`Dashboard.to_grafana` emits importable Grafana JSON
   (PromQL queries against the metrics exposed by :class:`MetricsCollector`).
2. **Live in-process view**: :class:`MetricsDashboard` binds panels to the running
   components, keeps short history for sparklines and renders a JSON snapshot,
   a terminal view or a self-contained HTML page. Components are duck-typed, so this
   module does not import the rest of the package.
"""

from __future__ import annotations

import asyncio
import html
import json
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

__all__ = ["PanelType", "Panel", "Row", "Dashboard", "MetricsDashboard", "build_default_dashboard"]

_SPARK = "▁▂▃▄▅▆▇█"
_COLORS = {"green": "#2e9e5b", "yellow": "#d9a400", "red": "#d64545", "blue": "#3b7dd8"}


class PanelType(str, Enum):
    STAT = "stat"
    GAUGE = "gauge"
    TIMESERIES = "timeseries"
    TABLE = "table"
    STATUS = "status"


@dataclass
class Panel:
    title: str
    type: PanelType
    binding: str  # key resolved by MetricsDashboard (e.g. "throughput_rps")
    promql: List[Tuple[str, str]] = field(default_factory=list)  # (expression, legend)
    unit: str = ""
    description: str = ""
    width: int = 6
    height: int = 6
    # ascending (value, color) steps; first step's value is ignored (base color)
    thresholds: List[Tuple[float, str]] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)  # for TABLE panels

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title, "type": self.type.value, "binding": self.binding,
            "promql": [{"expr": e, "legend": l} for e, l in self.promql], "unit": self.unit,
            "description": self.description, "width": self.width, "height": self.height,
            "thresholds": [{"value": v, "color": c} for v, c in self.thresholds],
        }


@dataclass
class Row:
    title: str
    panels: List[Panel] = field(default_factory=list)


@dataclass
class Dashboard:
    title: str
    uid: str
    rows: List[Row] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    refresh: str = "10s"
    time_range: str = "1h"

    def add_row(self, row: Row) -> "Dashboard":
        self.rows.append(row)
        return self

    def panels(self) -> List[Panel]:
        return [p for r in self.rows for p in r.panels]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title, "uid": self.uid, "tags": self.tags, "refresh": self.refresh,
            "time_range": self.time_range,
            "rows": [{"title": r.title, "panels": [p.to_dict() for p in r.panels]} for r in self.rows],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_grafana(self, datasource: str = "Prometheus") -> Dict[str, Any]:
        """Importable Grafana dashboard model (24-column grid, one collapsible row per Row)."""
        panels: List[Dict[str, Any]] = []
        panel_id, y = 1, 0
        for row in self.rows:
            panels.append({
                "id": panel_id, "type": "row", "title": row.title, "collapsed": False,
                "gridPos": {"h": 1, "w": 24, "x": 0, "y": y}, "panels": [],
            })
            panel_id += 1
            y += 1
            x, row_h = 0, 0
            for p in row.panels:
                if x + p.width > 24:
                    x, y, row_h = 0, y + row_h, 0
                panels.append(self._grafana_panel(p, panel_id, x, y, datasource))
                panel_id += 1
                x += p.width
                row_h = max(row_h, p.height)
            y += row_h
        return {
            "uid": self.uid, "title": self.title, "tags": self.tags, "schemaVersion": 39,
            "version": 1, "refresh": self.refresh, "time": {"from": f"now-{self.time_range}", "to": "now"},
            "timezone": "browser", "panels": panels,
        }

    @staticmethod
    def _grafana_panel(p: Panel, pid: int, x: int, y: int, ds: str) -> Dict[str, Any]:
        gtype = {"stat": "stat", "gauge": "gauge", "timeseries": "timeseries",
                 "table": "table", "status": "stat"}[p.type.value]
        steps = [{"color": "green", "value": None}]
        for value, color in p.thresholds[1:]:
            steps.append({"color": color, "value": value})
        if not p.thresholds:
            steps = [{"color": "blue", "value": None}]
        return {
            "id": pid, "type": gtype, "title": p.title, "description": p.description,
            "datasource": {"type": "prometheus", "uid": ds},
            "gridPos": {"h": p.height, "w": p.width, "x": x, "y": y},
            "targets": [
                {"refId": chr(65 + i), "expr": expr, "legendFormat": legend}
                for i, (expr, legend) in enumerate(p.promql)
            ],
            "fieldConfig": {
                "defaults": {"unit": p.unit or "short", "thresholds": {"mode": "absolute", "steps": steps}},
                "overrides": [],
            },
        }


def build_default_dashboard(service_name: str = "service") -> Dashboard:
    """RED + USE + errors dashboard matching the metric names this package exports."""
    t_lat = [(0, "green"), (0.5, "yellow"), (1.0, "red")]
    t_pct = [(0, "green"), (70, "yellow"), (90, "red")]
    dash = Dashboard(f"{service_name} - Service Overview", f"{service_name}-overview".lower().replace(" ", "-"),
                     tags=["observability", "red", "use"])
    dash.add_row(Row("Traffic & Latency (RED)", [
        Panel("Throughput", PanelType.TIMESERIES, "throughput_rps",
              [("sum(rate(http_requests_total[1m]))", "req/s")], "reqps", width=8),
        Panel("Latency p50/p95/p99", PanelType.TIMESERIES, "latency_p95_ms",
              [(f"histogram_quantile({q}, sum by (le) (rate(http_request_duration_seconds_bucket[5m])))", f"p{int(q*100)}")
               for q in (0.5, 0.95, 0.99)], "s", thresholds=t_lat, width=8),
        Panel("Error ratio (5xx)", PanelType.STAT, "error_ratio",
              [('sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))', "5xx")],
              "percentunit", thresholds=[(0, "green"), (0.01, "yellow"), (0.05, "red")], width=4),
        Panel("In flight", PanelType.STAT, "in_flight", [("sum(http_requests_in_flight)", "in flight")], width=4),
    ]))
    dash.add_row(Row("Saturation (USE)", [
        Panel("Process CPU", PanelType.GAUGE, "cpu_percent", [("process_cpu_percent", "process")],
              "percent", thresholds=t_pct, width=6),
        Panel("System memory", PanelType.GAUGE, "memory_percent", [("system_memory_used_percent", "system")],
              "percent", thresholds=t_pct, width=6),
        Panel("Process RSS", PanelType.TIMESERIES, "memory_rss_mb",
              [("process_resident_memory_bytes", "rss")], "bytes", width=6),
        Panel("Event loop lag", PanelType.TIMESERIES, "event_loop_lag_ms",
              [("event_loop_lag_seconds", "lag")], "s",
              thresholds=[(0, "green"), (0.05, "yellow"), (0.25, "red")], width=6),
    ]))
    dash.add_row(Row("Health & Errors", [
        Panel("Health", PanelType.STATUS, "health", [("min(health_check_status)", "health")], width=6),
        Panel("Top errors", PanelType.TABLE, "top_errors",
              [("topk(10, sum by (exception) (increase(errors_total[1h])))", "{{exception}}")],
              width=9, columns=["type", "count", "last_seen"]),
        Panel("Slowest operations", PanelType.TABLE, "slow_operations",
              [("topk(10, histogram_quantile(0.95, sum by (le, operation) "
                "(rate(operation_duration_seconds_bucket[5m]))))", "{{operation}}")],
              "s", width=9, columns=["operation", "count", "p95 ms"]),
    ]))
    return dash


class MetricsDashboard:
    """Live dashboard bound to running observability components (all optional)."""

    _HISTORY_BINDINGS = (
        "throughput_rps", "latency_p95_ms", "cpu_percent", "memory_percent",
        "memory_rss_mb", "event_loop_lag_ms", "error_ratio", "in_flight",
    )

    def __init__(
        self,
        definition: Optional[Dashboard] = None,
        *,
        service_name: str = "service",
        metrics: Any = None,
        health: Any = None,
        errors: Any = None,
        performance: Any = None,
        tracing: Any = None,
        history_size: int = 120,
    ) -> None:
        self.definition = definition or build_default_dashboard(service_name)
        self.service_name = service_name
        self.metrics, self.health, self.errors = metrics, health, errors
        self.performance, self.tracing = performance, tracing
        self._history: Dict[str, Deque[Tuple[float, float]]] = {
            k: deque(maxlen=history_size) for k in self._HISTORY_BINDINGS
        }
        self._task: Optional["asyncio.Task[None]"] = None

    # -- data resolution -----------------------------------------------------
    def _resolve(self, binding: str) -> Any:
        m, perf = self.metrics, self.performance
        sysm = (m.last_system if m is not None else {}) or {}
        try:
            if binding == "throughput_rps":
                return m.throughput() if m else None
            if binding == "error_ratio":
                return m.error_ratio() if m else None
            if binding == "in_flight":
                return m.in_flight.value() if m else None
            if binding.startswith("latency_"):
                key = binding.replace("latency_", "")  # p50_ms / p95_ms / p99_ms
                return m.latency_percentiles().get(key) if m else None
            if binding == "cpu_percent":
                return sysm.get("process_cpu_percent")
            if binding == "memory_percent":
                return sysm.get("system_memory_percent")
            if binding == "memory_rss_mb":
                rss = sysm.get("process_rss_bytes")
                return None if rss is None else rss / 1048576.0
            if binding == "event_loop_lag_ms":
                return perf.loop_lag_ms["last"] if perf else None
            if binding == "health":
                report = self.health.last_report() if self.health else None
                if report is None:
                    return None
                return {"status": report.status.value,
                        "checks": [{"name": c.name, "status": c.status.value, "message": c.message}
                                   for c in report.checks]}
            if binding == "top_errors":
                if not self.errors:
                    return []
                return [{"type": g.exc_type, "count": g.count, "last_seen": time.strftime("%H:%M:%S", time.localtime(g.last_seen))}
                        for g in self.errors.top_groups(10)]
            if binding == "slow_operations":
                if not perf:
                    return []
                return [{"operation": s.operation, "count": s.count, "p95 ms": round(s.p95_ms, 1)}
                        for s in perf.all_stats(limit=10)]
        except Exception:
            return None
        return None

    def sample(self) -> None:
        """Append current values of time-series bindings to the in-memory history."""
        now = time.time()
        for key, series in self._history.items():
            value = self._resolve(key)
            if isinstance(value, (int, float)):
                series.append((now, float(value)))

    async def start(self, interval: float = 5.0) -> None:
        if self._task is not None and not self._task.done():
            return

        async def loop() -> None:
            while True:
                self.sample()
                await asyncio.sleep(interval)

        self._task = asyncio.get_running_loop().create_task(loop(), name="obs-dashboard-sampler")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    # -- outputs -------------------------------------------------------------
    @staticmethod
    def _color(value: Any, thresholds: Sequence[Tuple[float, str]]) -> str:
        if not isinstance(value, (int, float)) or not thresholds:
            return "blue"
        color = thresholds[0][1]
        for limit, c in thresholds[1:]:
            if value >= limit:
                color = c
        return color

    def snapshot(self) -> Dict[str, Any]:
        """JSON-serialisable dashboard state: current value + history per panel."""
        self.sample()
        rows = []
        for row in self.definition.rows:
            panels = []
            for p in row.panels:
                value = self._resolve(p.binding)
                entry: Dict[str, Any] = {
                    "title": p.title, "type": p.type.value, "unit": p.unit, "value": value,
                    "color": self._color(value, p.thresholds),
                }
                if p.binding in self._history:
                    entry["history"] = [[round(t, 1), v] for t, v in self._history[p.binding]]
                panels.append(entry)
            rows.append({"title": row.title, "panels": panels})
        return {"title": self.definition.title, "generated_at": time.time(), "rows": rows}

    @staticmethod
    def _fmt(value: Any, unit: str) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, (int, float)):
            if unit == "percentunit":
                return f"{value * 100:.2f}%"
            if unit == "percent":
                return f"{value:.1f}%"
            suffix = {"reqps": " req/s", "bytes": " MB", "s": " ms"}.get(unit, "")
            return f"{value:,.2f}{suffix}" if abs(value) < 1000 else f"{value:,.0f}{suffix}"
        return str(value)

    @staticmethod
    def _sparkline(values: Sequence[float], width: int = 40) -> str:
        values = list(values)[-width:]
        if not values:
            return ""
        lo, hi = min(values), max(values)
        span = (hi - lo) or 1.0
        return "".join(_SPARK[int((v - lo) / span * (len(_SPARK) - 1))] for v in values)

    def render_text(self, width: int = 88) -> str:
        """Terminal-friendly rendering."""
        snap = self.snapshot()
        lines = ["=" * width, snap["title"].center(width), "=" * width]
        for row in snap["rows"]:
            lines += ["", f"## {row['title']}", "-" * width]
            for p in row["panels"]:
                if p["type"] == "table":
                    lines.append(f"{p['title']}:")
                    rows = p["value"] or []
                    if not rows:
                        lines.append("   (none)")
                    for r in rows[:10]:
                        lines.append("   " + "  ".join(str(v) for v in r.values()))
                elif p["type"] == "status":
                    status = (p["value"] or {}).get("status", "n/a") if isinstance(p["value"], dict) else "n/a"
                    lines.append(f"{p['title']:<26} {status.upper()}")
                    for c in (p["value"] or {}).get("checks", []) if isinstance(p["value"], dict) else []:
                        lines.append(f"   - {c['name']}: {c['status']} {c['message']}")
                else:
                    spark = self._sparkline([v for _, v in p.get("history", [])])
                    lines.append(f"{p['title']:<26} {self._fmt(p['value'], p['unit']):>14}  {spark}")
        return "\n".join(lines)

    def render_html(self, refresh_seconds: int = 5) -> str:
        """Self-contained HTML page (no external assets); embed behind your own auth."""
        snap = self.snapshot()
        esc = html.escape
        cards: List[str] = []
        for row in snap["rows"]:
            cards.append(f"<h2>{esc(row['title'])}</h2><div class='grid'>")
            for p in row["panels"]:
                body = ""
                if p["type"] == "table":
                    rows = p["value"] or []
                    if rows:
                        head = "".join(f"<th>{esc(str(k))}</th>" for k in rows[0])
                        trs = "".join("<tr>" + "".join(f"<td>{esc(str(v))}</td>" for v in r.values()) + "</tr>" for r in rows)
                        body = f"<table><tr>{head}</tr>{trs}</table>"
                    else:
                        body = "<div class='muted'>none</div>"
                elif p["type"] == "status":
                    val = p["value"] if isinstance(p["value"], dict) else {}
                    status = val.get("status", "n/a")
                    color = {"healthy": "green", "degraded": "yellow", "unhealthy": "red"}.get(status, "blue")
                    items = "".join(f"<li>{esc(c['name'])}: {esc(c['status'])} <span class='muted'>{esc(c['message'])}</span></li>"
                                    for c in val.get("checks", []))
                    body = f"<div class='big' style='color:{_COLORS[color]}'>{esc(status.upper())}</div><ul>{items}</ul>"
                else:
                    color = _COLORS.get(p["color"], _COLORS["blue"])
                    body = f"<div class='big' style='color:{color}'>{esc(self._fmt(p['value'], p['unit']))}</div>"
                    hist = [v for _, v in p.get("history", [])]
                    if len(hist) > 1:
                        lo, hi = min(hist), max(hist)
                        span = (hi - lo) or 1.0
                        pts = " ".join(f"{i * 200 / (len(hist) - 1):.1f},{40 - (v - lo) / span * 36 - 2:.1f}"
                                       for i, v in enumerate(hist))
                        body += f"<svg viewBox='0 0 200 40' preserveAspectRatio='none'><polyline fill='none' stroke='{color}' stroke-width='1.5' points='{pts}'/></svg>"
                cards.append(f"<div class='card'><div class='t'>{esc(p['title'])}</div>{body}</div>")
            cards.append("</div>")
        return (
            "<!doctype html><html><head><meta charset='utf-8'>"
            f"<meta http-equiv='refresh' content='{int(refresh_seconds)}'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{esc(snap['title'])}</title><style>"
            ":root{color-scheme:light dark;--bg:#fff;--fg:#1c1f24;--card:#f4f5f7;--muted:#6b7280}"
            "@media (prefers-color-scheme:dark){:root{--bg:#14161a;--fg:#e6e8eb;--card:#1e2127;--muted:#9aa1ab}}"
            "body{font-family:system-ui,sans-serif;margin:0;padding:16px;background:var(--bg);color:var(--fg)}"
            "h1{font-size:20px}h2{font-size:15px;margin:20px 0 8px}"
            ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px}"
            ".card{background:var(--card);border-radius:8px;padding:12px;overflow-x:auto}"
            ".t{font-size:12px;color:var(--muted);margin-bottom:4px}.big{font-size:26px;font-weight:600}"
            ".muted{color:var(--muted)}svg{width:100%;height:40px}table{font-size:12px;border-collapse:collapse;width:100%}"
            "td,th{padding:2px 6px;text-align:left}ul{padding-left:16px;font-size:12px;margin:6px 0 0}"
            f"</style></head><body><h1>{esc(snap['title'])}</h1>{''.join(cards)}</body></html>"
        )

    def to_grafana_json(self, datasource: str = "Prometheus", indent: int = 2) -> str:
        return json.dumps(self.definition.to_grafana(datasource), indent=indent)
