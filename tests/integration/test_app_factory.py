import pytest


pytestmark = pytest.mark.integration


def test_configured_app_factory_returns_application(app_instance):
    assert app_instance is not None, "App factory returned None"
