import time
import pytest


pytestmark = pytest.mark.performance


def test_configured_target_finishes_within_smoke_budget(perf_callable):
    """
    Basic single-call smoke timing only. This is NOT a load test or benchmark.
    Configure ZENTRAX_PERF_MAX_SECONDS to set the budget (default 10 seconds).
    """
    import os

    budget = float(os.getenv("ZENTRAX_PERF_MAX_SECONDS", "10"))
    started = time.perf_counter()
    result = perf_callable()
    elapsed = time.perf_counter() - started
    assert result is not None, "Performance target returned None"
    assert elapsed <= budget, f"Target took {elapsed:.3f}s; budget is {budget:.3f}s"
