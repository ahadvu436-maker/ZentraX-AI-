"""Small HTTP helpers; no external HTTP client dependency is required."""
from __future__ import annotations

from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def get_status(url: str, timeout: float = 5.0) -> int:
    request = Request(url, headers={"User-Agent": "ZentraX-Test-Suite/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return int(response.status)
    except HTTPError as exc:
        return int(exc.code)
    except URLError as exc:
        raise ConnectionError(f"Could not reach {url}: {exc}") from exc
