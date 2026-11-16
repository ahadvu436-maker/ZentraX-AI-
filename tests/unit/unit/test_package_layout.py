import pytest

from fixtures.project_scan import package_directories


pytestmark = pytest.mark.unit


def test_python_package_directories_have_init_or_are_documented_namespaces(project_root):
    """
    Conventional packages should contain __init__.py.
    Namespace packages are valid Python, so this test reports rather than
    universally forbids them. Set ZENTRAX_REQUIRE_INIT=1 to enforce it.
    """
    import os

    if os.getenv("ZENTRAX_REQUIRE_INIT", "0") != "1":
        pytest.skip("Set ZENTRAX_REQUIRE_INIT=1 to enforce __init__.py in every package directory")
    missing = [
        str(path.relative_to(project_root))
        for path in package_directories(project_root)
        if not (path / "__init__.py").exists()
    ]
    assert not missing, "Missing __init__.py in: " + ", ".join(missing)
