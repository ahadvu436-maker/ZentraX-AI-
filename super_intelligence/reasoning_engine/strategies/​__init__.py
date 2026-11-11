"""strategies - the three classic reasoning styles, each an IntelligenceModule.

    deductive  : general truths  -> specific truths   (certain, if premises hold)
    inductive  : specific cases  -> general theory    (probable)
    abductive  : incomplete data -> best explanation  (plausible)
"""
from typing import Any, Dict, Type

from intelligence_interface import IntelligenceModule

from .abductive import AbductiveStrategy
from .deductive import DeductiveStrategy
from .inductive import InductiveStrategy

STRATEGIES: Dict[str, Type[IntelligenceModule]] = {
    "deductive": DeductiveStrategy,
    "inductive": InductiveStrategy,
    "abductive": AbductiveStrategy,
}


def get_strategy(name: str, **kwargs: Any) -> IntelligenceModule:
    try:
        return STRATEGIES[name](**kwargs)
    except KeyError:
        raise ValueError(f"unknown strategy '{name}', choose from {list(STRATEGIES)}") from None


__all__ = ["DeductiveStrategy", "InductiveStrategy", "AbductiveStrategy", "STRATEGIES", "get_strategy"]
