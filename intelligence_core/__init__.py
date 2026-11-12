"""01_core_intelligence - command center of the system.

    intelligence_core    : orchestrates everything, makes the central decision
    cognitive_engine     : raw data -> higher-level insight
    context_manager      : live work context, history, environment
    decision_engine      : picks the best and safest option
    intelligence_router  : sends each request to the right subsystem
    intelligence_interface : contracts shared with every module and external system

The folder name starts with a digit, so import it with
importlib.import_module("01_core_intelligence"), or simply put this folder on
sys.path and import the modules directly. This file does the second part
automatically, so the modules can import each other by plain name.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from intelligence_interface import (  # noqa: E402
    ExternalGateway, ExternalSystem, InMemoryMemory, IntelligenceModule,
    MemoryProvider, ReasoningProvider, Result, Signal,
)
from intelligence_core import IntelligenceCore  # noqa: E402
from cognitive_engine import CognitiveEngine  # noqa: E402
from context_manager import ContextManager  # noqa: E402
from decision_engine import DecisionEngine  # noqa: E402
from intelligence_router import IntelligenceRouter  # noqa: E402


def create_default_core(**decision_kwargs) -> IntelligenceCore:
    """IntelligenceCore with all four modules registered.

    decision_kwargs go to DecisionEngine, e.g. max_risk=0.3, blocked_actions=["execute"].
    """
    core = IntelligenceCore()
    core.register("context", ContextManager())
    core.register("router", IntelligenceRouter())
    core.register("cognitive", CognitiveEngine())
    core.register("decision", DecisionEngine(**decision_kwargs))
    return core


__all__ = [
    "IntelligenceCore", "CognitiveEngine", "ContextManager", "DecisionEngine", "IntelligenceRouter",
    "IntelligenceModule", "ExternalSystem", "MemoryProvider", "ReasoningProvider", "ExternalGateway",
    "InMemoryMemory", "Signal", "Result", "create_default_core",
]
