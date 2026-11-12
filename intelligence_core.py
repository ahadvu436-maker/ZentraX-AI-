"""intelligence_core.py - command center of 01_core_intelligence.

Collects signals from all sub-modules and makes the central decision.
It is the only module that knows the others; they only know the interface.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from intelligence_interface import IntelligenceModule, Result, Signal

log = logging.getLogger("intelligence_core")

# Roles the core orchestrates.
ROLES = ("router", "context", "cognitive", "decision")


class IntelligenceCore:
    def __init__(self) -> None:
        self._modules: Dict[str, IntelligenceModule] = {}
        self._signals: list[Signal] = []  # recent signals from sub-modules
        self._max_signals = 500

    # ---- registration -------------------------------------------------
    def register(self, role: str, module: IntelligenceModule) -> None:
        if role not in ROLES:
            raise ValueError(f"Unknown role '{role}', expected one of {ROLES}")
        self._modules[role] = module
        log.info("registered %s as %s", module.name, role)

    def _require(self, role: str) -> IntelligenceModule:
        if role not in self._modules:
            raise RuntimeError(f"Module for role '{role}' is not registered")
        return self._modules[role]

    # ---- signal collection -------------------------------------------
    def _collect(self, source: str, payload: Any, kind: str = "generic") -> Signal:
        sig = Signal(source=source, payload=payload, kind=kind)
        self._signals.append(sig)
        del self._signals[: -self._max_signals]
        return sig

    def collect_states(self) -> Dict[str, Dict[str, Any]]:
        """Gather live state and health from every registered module."""
        return {
            role: {"state": m.get_state(), "healthy": m.health_check()}
            for role, m in self._modules.items()
        }

    # ---- central pipeline --------------------------------------------
    def handle(self, request: Any) -> Result:
        """request -> context -> route -> think -> decide -> remember."""
        try:
            context = self._require("context")
            router = self._require("router")
            cognitive = self._require("cognitive")
            decision = self._require("decision")

            # 1. Attach the current context to the request.
            ctx_res = context.process(self._collect("core", request, "request"))
            enriched = {"request": request, "context": ctx_res.data}

            # 2. Route to the right subsystem.
            route_res = router.process(self._collect("core", enriched, "route"))
            if not route_res.ok:
                return self._safe_fallback(f"routing failed: {route_res.error}")
            enriched["route"] = route_res.data

            # 3. Turn raw data into higher-level understanding.
            think_res = cognitive.process(self._collect("router", enriched, "think"))
            if not think_res.ok:
                return self._safe_fallback(f"cognition failed: {think_res.error}")
            enriched["insight"] = think_res.data

            # 4. Pick the best and safest option.
            dec_res = decision.process(self._collect("cognitive", enriched, "decide"))
            if not dec_res.ok:
                return self._safe_fallback(f"decision failed: {dec_res.error}")

            # 5. Feed the outcome back so context stays live.
            context.process(self._collect("decision", dec_res.data, "outcome"))
            return dec_res

        except Exception as exc:  # the core must never crash the system
            log.exception("core pipeline error")
            return self._safe_fallback(str(exc))

    @staticmethod
    def _safe_fallback(reason: str) -> Result:
        log.warning("fallback: %s", reason)
        return Result(ok=False, error=reason, data={"action": "no_op"})
