import os
import pytest

from fixtures.http_helpers import get_status


pytestmark = pytest.mark.system


def test_deployed_health_endpoint_returns_success():
    url = os.getenv("ZENTRAX_HEALTH_URL", "").strip()
    if not url:
        pytest.skip("Set ZENTRAX_HEALTH_URL to enable deployed health check")
    status = get_status(url, timeout=float(os.getenv("ZENTRAX_HTTP_TIMEOUT", "5")))
    assert 200 <= status < 300, f"Health endpoint returned HTTP {status}"
