import pytest


pytestmark = pytest.mark.system


def test_app_exposes_expected_health_contract(app_instance):
    """
    Optional generic contract:
    - FastAPI/Starlette app: should expose .routes
    - WSGI/ASGI app: may expose callable interface
    This deliberately does not assume ZentraX's exact framework.
    """
    has_routes = hasattr(app_instance, "routes")
    is_callable = callable(app_instance)
    assert has_routes or is_callable, (
        "App should expose framework routes or be callable as an application"
    )
